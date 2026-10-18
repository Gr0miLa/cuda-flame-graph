#!/usr/bin/env python3

"""Build an exact SASS index and enrich folded PC-sampling stacks."""

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
PC_FRAME_RE = re.compile(r"^0x([0-9a-f]+)$")
WEIGHT_RE = re.compile(r"^[1-9][0-9]*$")


class SassError(RuntimeError):
    """Base error for SASS indexing failures."""


class SassParseError(SassError):
    """Raised when structured cuobjdump output is internally inconsistent."""


class ToolExecutionError(SassError):
    """Raised when an external CUDA binary utility cannot be executed."""


class FoldedParseError(SassError):
    """Raised when a folded row cannot be enriched safely."""


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


@dataclass(frozen=True)
class FoldedRow:
    frames: tuple[str, ...]
    weight: int
    pc_offset: int


@dataclass
class EnrichmentStats:
    rows_read: int = 0
    total_weight: int = 0
    resolved_rows: int = 0
    resolved_weight: int = 0
    unresolved_rows: int = 0
    unresolved_weight: int = 0
    ambiguous_functions: int = 0
    architecture_mismatches: int = 0
    malformed_rows: int = 0
    unique_kernel_pc_pairs: set[tuple[str, int]] = field(default_factory=set)


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


def parse_folded_line(line: str, line_number: int) -> FoldedRow:
    stripped = line.rstrip("\r\n")
    if not stripped:
        raise FoldedParseError(f"line {line_number}: empty folded row")
    try:
        stack, weight_text = stripped.rsplit(maxsplit=1)
    except ValueError as error:
        raise FoldedParseError(
            f"line {line_number}: missing stack or weight"
        ) from error
    if not WEIGHT_RE.fullmatch(weight_text):
        raise FoldedParseError(
            f"line {line_number}: weight must be a positive decimal integer"
        )

    frames = tuple(stack.split(";"))
    if len(frames) < 3 or any(not frame for frame in frames):
        raise FoldedParseError(
            f"line {line_number}: expected non-empty kernel;stall;PC tail"
        )
    pc_match = PC_FRAME_RE.fullmatch(frames[-1])
    if pc_match is None:
        raise FoldedParseError(
            f"line {line_number}: last frame must be lowercase hexadecimal PC"
        )
    return FoldedRow(frames, int(weight_text, 10), int(pc_match.group(1), 16))


def enrich_folded_lines(
    lines: Iterable[str], index: SassIndex, architecture: str
) -> tuple[list[str], EnrichmentStats]:
    if architecture not in index.architectures:
        available = ", ".join(index.architectures) or "none"
        raise SassError(
            f"architecture {architecture} is absent; available: {available}"
        )

    output: list[str] = []
    stats = EnrichmentStats()
    for line_number, line in enumerate(lines, start=1):
        try:
            row = parse_folded_line(line, line_number)
        except FoldedParseError:
            stats.malformed_rows += 1
            raise

        kernel = row.frames[-3]
        resolution = index.resolve(architecture, kernel, row.pc_offset)

        # A second pass would otherwise interpret the mnemonic as the kernel.
        if resolution.status == "unknown_function" and len(row.frames) >= 4:
            previous = index.resolve(architecture, row.frames[-4], row.pc_offset)
            if previous.status == "resolved" and previous.mnemonic == kernel:
                raise FoldedParseError(
                    f"line {line_number}: input appears to be already enriched"
                )

        stats.rows_read += 1
        stats.total_weight += row.weight
        stats.unique_kernel_pc_pairs.add((kernel, row.pc_offset))
        if resolution.status == "resolved":
            frames = row.frames[:-2] + (resolution.mnemonic,) + row.frames[-2:]
            stats.resolved_rows += 1
            stats.resolved_weight += row.weight
        else:
            frames = row.frames
            stats.unresolved_rows += 1
            stats.unresolved_weight += row.weight
            if resolution.status == "ambiguous_function":
                stats.ambiguous_functions += 1
            elif resolution.status == "unknown_architecture":
                stats.architecture_mismatches += 1
        output.append(f"{';'.join(frames)} {row.weight}\n")

    if stats.resolved_rows + stats.unresolved_rows != stats.rows_read:
        raise SassError("internal error: folded row accounting mismatch")
    if stats.resolved_weight + stats.unresolved_weight != stats.total_weight:
        raise SassError("internal error: folded weight accounting mismatch")
    return output, stats


def enrich_folded_file(
    input_path: Path, output_path: Path, index: SassIndex, architecture: str
) -> EnrichmentStats:
    if input_path.resolve() == output_path.resolve():
        raise SassError("input and output paths must be different")
    if not input_path.is_file():
        raise SassError(f"folded input does not exist: {input_path}")

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    enriched, stats = enrich_folded_lines(lines, index, architecture)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(enriched), encoding="utf-8")
    return stats


def print_enrichment_stats(stats: EnrichmentStats) -> None:
    row_percent = (
        100.0 * stats.resolved_rows / stats.rows_read if stats.rows_read else 0.0
    )
    weight_percent = (
        100.0 * stats.resolved_weight / stats.total_weight
        if stats.total_weight
        else 0.0
    )
    print(f"rows read: {stats.rows_read}", file=sys.stderr)
    print(
        f"unique kernel/PC pairs: {len(stats.unique_kernel_pc_pairs)}",
        file=sys.stderr,
    )
    print(
        f"resolved: rows={stats.resolved_rows}, weight={stats.resolved_weight}",
        file=sys.stderr,
    )
    print(
        f"unresolved: rows={stats.unresolved_rows}, weight={stats.unresolved_weight}",
        file=sys.stderr,
    )
    print(f"ambiguous functions: {stats.ambiguous_functions}", file=sys.stderr)
    print(
        f"architecture mismatches: {stats.architecture_mismatches}",
        file=sys.stderr,
    )
    print(f"malformed rows: {stats.malformed_rows}", file=sys.stderr)
    print(f"resolution by rows: {row_percent:.3f}%", file=sys.stderr)
    print(f"resolution by weight: {weight_percent:.3f}%", file=sys.stderr)


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect an exact SASS index from a CUDA binary."
    )
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--arch", required=True, help="CUDA architecture, e.g. sm_75")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--demangler", default="cu++filt")
    parser.add_argument("--input", type=Path, help="raw folded input")
    parser.add_argument("--output", type=Path, help="enriched folded output")
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
        if (args.input is None) != (args.output is None):
            raise SassError("--input and --output must be specified together")
        index = build_sass_index(args.binary, args.cuobjdump, args.demangler)
        functions = index.functions_for_architecture(args.arch)
        if not functions:
            available = ", ".join(index.architectures) or "none"
            raise SassError(
                f"architecture {args.arch} is absent; available: {available}"
            )
        if args.input is not None:
            stats = enrich_folded_file(args.input, args.output, index, args.arch)
            print_enrichment_stats(stats)
            return 0
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
