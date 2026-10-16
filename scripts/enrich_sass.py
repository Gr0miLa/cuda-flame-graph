#!/usr/bin/env python3

"""Build an exact SASS instruction index from cuobjdump output.

Folded-stack enrichment is intentionally not implemented in this stage.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


ARCH_RE = re.compile(r"^\s*arch\s*=\s*(sm_[0-9]+)\s*$")
FUNCTION_RE = re.compile(r"^\s*Function\s*:\s*(\S.*?)\s*$")
INSTRUCTION_RE = re.compile(
    r"^\s*/\*([0-9a-fA-F]+)\*/\s+"
    r"(?:@!?[A-Za-z0-9_.]+\s+)?"
    r"([A-Za-z][A-Za-z0-9_.]*)\b"
)


class SassError(RuntimeError):
    """Base error for SASS indexing failures."""


class SassParseError(SassError):
    """Raised when structured cuobjdump output is internally inconsistent."""


class ToolExecutionError(SassError):
    """Raised when an external CUDA binary utility cannot be executed."""


@dataclass
class SassFunction:
    architecture: str
    original_name: str
    instructions: dict[int, str] = field(default_factory=dict)
    demangled_name: str = ""
    short_name: str = ""


@dataclass(frozen=True)
class Resolution:
    status: str
    mnemonic: str | None = None
    matched_function: str | None = None


class SassIndex:
    def __init__(self, functions: Iterable[SassFunction]):
        self.functions = list(functions)

    @property
    def architectures(self) -> tuple[str, ...]:
        return tuple(sorted({function.architecture for function in self.functions}))

    def functions_for_architecture(self, architecture: str) -> list[SassFunction]:
        return [
            function
            for function in self.functions
            if function.architecture == architecture
        ]

    def instruction_count(self, architecture: str | None = None) -> int:
        functions = (
            self.functions
            if architecture is None
            else self.functions_for_architecture(architecture)
        )
        return sum(len(function.instructions) for function in functions)

    def resolve(self, architecture: str, function_name: str, pc_offset: int) -> Resolution:
        functions = self.functions_for_architecture(architecture)
        if not functions:
            return Resolution("unknown_architecture")

        exact_matches = [
            function
            for function in functions
            if function_name in {function.original_name, function.demangled_name}
        ]
        candidates = exact_matches
        if not candidates:
            candidates = [
                function for function in functions if function.short_name == function_name
            ]

        if not candidates:
            return Resolution("unknown_function")
        if len(candidates) != 1:
            return Resolution("ambiguous_function")

        function = candidates[0]
        mnemonic = function.instructions.get(pc_offset)
        if mnemonic is None:
            return Resolution("unknown_pc", matched_function=function.original_name)
        return Resolution("resolved", mnemonic, function.original_name)


def parse_cuobjdump_sass(output: str) -> SassIndex:
    functions: list[SassFunction] = []
    architecture: str | None = None
    current_function: SassFunction | None = None

    for line_number, line in enumerate(output.splitlines(), start=1):
        architecture_match = ARCH_RE.match(line)
        if architecture_match:
            architecture = architecture_match.group(1)
            current_function = None
            continue

        function_match = FUNCTION_RE.match(line)
        if function_match:
            if architecture is None:
                raise SassParseError(
                    f"line {line_number}: function appears before architecture"
                )
            current_function = SassFunction(
                architecture=architecture,
                original_name=function_match.group(1),
            )
            functions.append(current_function)
            continue

        instruction_match = INSTRUCTION_RE.match(line)
        if instruction_match:
            if current_function is None:
                raise SassParseError(
                    f"line {line_number}: instruction appears before function"
                )
            pc_offset = int(instruction_match.group(1), 16)
            mnemonic = instruction_match.group(2)
            previous = current_function.instructions.get(pc_offset)
            if previous is not None and previous != mnemonic:
                raise SassParseError(
                    f"line {line_number}: conflicting instructions at "
                    f"{current_function.original_name}+0x{pc_offset:x}"
                )
            current_function.instructions[pc_offset] = mnemonic

    if not functions:
        raise SassParseError("cuobjdump output contains no SASS functions")
    if not any(function.instructions for function in functions):
        raise SassParseError("cuobjdump output contains no SASS instructions")
    return SassIndex(functions)


def _short_function_name(demangled_name: str) -> str:
    name_without_parameters = demangled_name.split("(", maxsplit=1)[0].strip()
    return name_without_parameters.rsplit("::", maxsplit=1)[-1]


def apply_demangled_names(
    index: SassIndex, demangled_names: dict[str, str]
) -> SassIndex:
    for function in index.functions:
        demangled_name = demangled_names.get(
            function.original_name, function.original_name
        ).strip()
        function.demangled_name = demangled_name
        function.short_name = _short_function_name(demangled_name)
    return index


def demangle_function_names(
    names: Sequence[str], tool: str = "cu++filt"
) -> dict[str, str]:
    unique_names = list(dict.fromkeys(names))
    if not unique_names:
        return {}

    try:
        result = subprocess.run(
            [tool],
            input="\n".join(unique_names) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ToolExecutionError(f"demangler not found: {tool}") from error

    if result.returncode != 0:
        detail = result.stderr.strip() or "no error details"
        raise ToolExecutionError(
            f"{tool} failed with exit code {result.returncode}: {detail}"
        )

    demangled = result.stdout.splitlines()
    if len(demangled) != len(unique_names):
        raise ToolExecutionError(
            f"{tool} returned {len(demangled)} names for {len(unique_names)} inputs"
        )
    return dict(zip(unique_names, demangled, strict=True))


def run_cuobjdump(binary: Path, tool: str = "cuobjdump") -> str:
    if not binary.is_file():
        raise ToolExecutionError(f"binary does not exist: {binary}")
    try:
        result = subprocess.run(
            [tool, "-sass", str(binary)],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ToolExecutionError(f"cuobjdump not found: {tool}") from error

    if result.returncode != 0:
        detail = result.stderr.strip() or "no error details"
        raise ToolExecutionError(
            f"{tool} failed with exit code {result.returncode}: {detail}"
        )
    return result.stdout


def build_sass_index(
    binary: Path,
    cuobjdump: str = "cuobjdump",
    demangler: str = "cu++filt",
) -> SassIndex:
    index = parse_cuobjdump_sass(run_cuobjdump(binary, cuobjdump))
    names = [function.original_name for function in index.functions]
    return apply_demangled_names(index, demangle_function_names(names, demangler))


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect an exact SASS index from a CUDA binary."
    )
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="CUDA architecture, e.g. sm_75")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--demangler", default="cu++filt")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="maximum instruction rows printed per function",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        index = build_sass_index(args.binary, args.cuobjdump, args.demangler)
        functions = index.functions_for_architecture(args.arch)
        if not functions:
            available = ", ".join(index.architectures) or "none"
            raise SassError(
                f"architecture {args.arch} is absent; available: {available}"
            )
    except SassError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"binary: {args.binary}")
    print(f"architectures: {', '.join(index.architectures)}")
    print(f"selected architecture: {args.arch}")
    print(f"functions: {len(functions)}")
    print(f"instructions: {index.instruction_count(args.arch)}")
    for function in functions:
        print(
            f"function: {function.original_name} -> {function.demangled_name} "
            f"({len(function.instructions)} instructions)"
        )
        for pc_offset, mnemonic in sorted(function.instructions.items())[
            : max(args.limit, 0)
        ]:
            print(f"  0x{pc_offset:x}\t{mnemonic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
