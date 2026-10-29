#!/usr/bin/env bash

set -u
set -o pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${project_dir}/results/experiment"
architecture="sm_75"
repeats=5
cuprof_path="${project_dir}/build/cuprof"
flamegraph_path="${project_dir}/../FlameGraph/flamegraph.pl"
python_path="${PYTHON_BIN:-python3}"
modes=(min low mid exp6 exp7 exp8 exp9 exp10 exp12 exp14 exp16 high max)

usage() {
    printf '%s\n' \
        "Запуск:" \
        "  scripts/run_sampling_experiment.sh [опции]" \
        "" \
        "Опции:" \
        "  --output ПАПКА     Куда сохранить эксперимент" \
        "  --arch АРХИТЕКТУРА Например sm_75" \
        "  --repeats ЧИСЛО    Число запусков каждого варианта (по умолчанию 5)" \
        "  --cuprof ПУТЬ      Путь к cuprof" \
        "  --flamegraph ПУТЬ  Путь к flamegraph.pl" \
        "  -h, --help         Показать эту справку"
}

fail() {
    printf 'Ошибка: %s\n' "$1" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || fail "после $1 нужно указать значение"
}

median() {
    sort -n "$1" | awk '
        { values[NR] = $1 }
        END {
            if (NR == 0) exit 1
            if (NR % 2) print values[(NR + 1) / 2]
            else printf "%.6f\n", (values[NR / 2] + values[NR / 2 + 1]) / 2
        }
    '
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            require_value "$@"
            output_dir="$2"
            shift 2
            ;;
        --arch)
            require_value "$@"
            architecture="$2"
            shift 2
            ;;
        --repeats)
            require_value "$@"
            repeats="$2"
            shift 2
            ;;
        --cuprof)
            require_value "$@"
            cuprof_path="$2"
            shift 2
            ;;
        --flamegraph)
            require_value "$@"
            flamegraph_path="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "неизвестная опция $1"
            ;;
    esac
done

[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || fail "--repeats должен быть положительным числом"
[[ -x "$cuprof_path" ]] || fail "не найден cuprof: $cuprof_path"
[[ -x "$flamegraph_path" ]] || fail "не найден flamegraph.pl: $flamegraph_path"
[[ -f "${project_dir}/scripts/enrich_sass.py" ]] || fail "не найден enrich_sass.py"
command -v "$python_path" >/dev/null 2>&1 || fail "не найден Python: $python_path"
command -v /usr/bin/time >/dev/null 2>&1 || fail "не найден /usr/bin/time"

if [[ -d "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -print -quit)" ]]; then
    fail "папка уже содержит файлы: $output_dir"
fi
mkdir -p "$output_dir" || fail "не удалось создать папку: $output_dir"

temp_dir="$(mktemp -d)" || fail "не удалось создать временную папку"
trap 'rm -rf -- "$temp_dir"' EXIT

summary_file="${output_dir}/summary.tsv"
printf 'sample\tmode\tmedian_seconds\tslowdown\n' > "$summary_file"
quality_file="${output_dir}/quality.tsv"
printf 'sample\tmode\tcycles\tpc_records\tdropped_percent\tfolded_rows\n' > "$quality_file"

samples_found=0
for source_dir in "${project_dir}"/samples/*; do
    [[ -f "${source_dir}/CMakeLists.txt" ]] || continue
    sample="$(basename "$source_dir")"
    binary="${project_dir}/build/samples/${sample}/${sample}"
    [[ -x "$binary" ]] || fail "sample не собран: $binary"
    samples_found=$((samples_found + 1))

    sample_dir="${output_dir}/${sample}"
    folded_dir="${sample_dir}/folded"
    stderr_dir="${sample_dir}/stderr"
    svg_dir="${sample_dir}/svg"
    mkdir -p "$folded_dir" "$stderr_dir" "$svg_dir"

    timings_file="${stderr_dir}/timings.tsv"
    printf 'mode\trun\tseconds\n' > "$timings_file"

    printf '[%s] baseline\n' "$sample"
    baseline_values="${temp_dir}/${sample}_baseline.txt"
    : > "$baseline_values"
    for ((run = 1; run <= repeats; ++run)); do
        time_file="${temp_dir}/time.txt"
        /usr/bin/time -f '%e' -o "$time_file" "$binary" \
            >"${temp_dir}/program.stdout" 2>"${temp_dir}/program.stderr" || \
            fail "$sample: обычный запуск $run завершился с ошибкой"
        seconds="$(<"$time_file")"
        printf 'baseline\t%d\t%s\n' "$run" "$seconds" >> "$timings_file"
        printf '%s\n' "$seconds" >> "$baseline_values"
    done
    baseline_median="$(median "$baseline_values")" || fail "$sample: не удалось вычислить baseline"
    printf '%s\tbaseline\t%s\t1.000\n' "$sample" "$baseline_median" >> "$summary_file"

    for mode in "${modes[@]}"; do
        printf '[%s] %s\n' "$sample" "$mode"
        period_values="${temp_dir}/${sample}_${mode}.txt"
        : > "$period_values"
        raw_folded="${folded_dir}/${mode}.folded"
        profiler_stderr="${stderr_dir}/${mode}.stderr"

        sampling_option=(--sampling-period "$mode")
        if [[ "$mode" == exp* ]]; then
            sampling_option=(--sampling-period-exp "${mode#exp}")
        fi

        for ((run = 1; run <= repeats; ++run)); do
            time_file="${temp_dir}/time.txt"
            if [[ "$run" -eq 1 ]]; then
                stdout_file="$raw_folded"
                stderr_file="$profiler_stderr"
            else
                stdout_file="${temp_dir}/profile.folded"
                stderr_file="${temp_dir}/profile.stderr"
            fi

            /usr/bin/time -f '%e' -o "$time_file" \
                "$cuprof_path" --stall-mode "${sampling_option[@]}" "$binary" \
                >"$stdout_file" 2>"$stderr_file" || \
                fail "$sample/$mode: запуск $run завершился с ошибкой"
            seconds="$(<"$time_file")"
            printf '%s\t%d\t%s\n' "$mode" "$run" "$seconds" >> "$timings_file"
            printf '%s\n' "$seconds" >> "$period_values"
        done

        period_median="$(median "$period_values")" || fail "$sample/$mode: нет времени"
        slowdown="$(awk -v prof="$period_median" -v base="$baseline_median" \
            'BEGIN { if (base == 0) print "n/a"; else printf "%.3f", prof / base }')"
        printf '%s\t%s\t%s\t%s\n' \
            "$sample" "$mode" "$period_median" "$slowdown" >> "$summary_file"

        cycles="$(sed -n 's/^PC Sampling period cycles: //p' "$profiler_stderr")"
        pc_records="$(sed -n 's/^PC Sampling records: //p' "$profiler_stderr")"
        dropped_percent="$(sed -n \
            's/^PC Sampling dropped samples: .* (\([0-9.]*%\))$/\1/p' \
            "$profiler_stderr")"
        folded_rows="$(wc -l < "$raw_folded")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$sample" "$mode" "$cycles" "$pc_records" \
            "$dropped_percent" "$folded_rows" >> "$quality_file"

        # При max короткий kernel может закончиться до первого аппаратного
        # наблюдения. Пустой folded в этом случае является результатом опыта.
        if [[ ! -s "$raw_folded" ]]; then
            printf '%s\n' "PC Sampling не собрал ни одного стека" \
                > "${stderr_dir}/${mode}_sass.stderr"
            continue
        fi

        sass_folded="${folded_dir}/${mode}_sass.folded"
        sass_stderr="${stderr_dir}/${mode}_sass.stderr"
        "$python_path" "${project_dir}/scripts/enrich_sass.py" \
            --input "$raw_folded" \
            --binary "$binary" \
            --arch "$architecture" \
            --output "$sass_folded" \
            >/dev/null 2>"$sass_stderr" || \
            fail "$sample/$mode: не удалось добавить SASS"

        "$flamegraph_path" --hash "$sass_folded" > "${svg_dir}/${mode}.svg" || \
            fail "$sample/$mode: не удалось создать SVG"
    done
done

[[ "$samples_found" -gt 0 ]] || fail "не найдено ни одного sample"

report_file="${output_dir}/README.md"
{
    printf '# Эксперимент с периодами PC Sampling\n\n'
    printf 'Каждый вариант запускался %d раз. В таблице указана медиана.\n\n' "$repeats"
    printf '`expN` означает пользовательский период 2^N GPU-циклов.\n\n'
    printf '| Sample | Режим | Медиана, секунд | Замедление |\n'
    printf '|---|---|---:|---:|\n'
    while IFS=$'\t' read -r sample mode seconds slowdown; do
        [[ "$sample" == "sample" ]] && continue
        printf '| %s | %s | %s | %s |\n' "$sample" "$mode" "$seconds" "$slowdown"
    done < "$summary_file"
    printf '\n## Качество профиля\n\n'
    printf '| Sample | Режим | Циклы | PC records | Потеряно | Folded-строк |\n'
    printf '|---|---|---:|---:|---:|---:|\n'
    while IFS=$'\t' read -r sample mode cycles records dropped rows; do
        [[ "$sample" == "sample" ]] && continue
        printf '| %s | %s | %s | %s | %s | %s |\n' \
            "$sample" "$mode" "$cycles" "$records" "$dropped" "$rows"
    done < "$quality_file"
    printf '\n`baseline` — запуск без профилировщика. Замедление считается как '
    printf '`медиана режима / медиана baseline`.\n\n'
    printf 'Время включает запуск процесса, CUDA initialization и сбор профиля. '
    printf 'SASS post-processing и создание SVG во время не входят.\n\n'
    printf 'Для коротких samples постоянные затраты запуска могут быть больше '
    printf 'стоимости самого kernel. Поэтому точнее всего влияние периода видно '
    printf 'на более долгих вычислениях.\n'
} > "$report_file"

printf '%s\n' \
    "Эксперимент завершён." \
    "Время и slowdown: $summary_file" \
    "Качество профиля: $quality_file" \
    "Краткий отчёт: $report_file" \
    "Количество повторов: $repeats"
