# cuda-flame-graph
Это профилировщик для CUDA-программ. Он собирает CPU-контекст запуска кернела,
GPU stall-сэмплы и делает folded-файл для FlameGraph.

Основной режим сейчас - stall mode. В графе получается примерно такая цепочка:
```text
CPU стек -> cudaLaunchKernel -> kernel -> SASS инструкция -> причина простоя -> PC
```

## Флаги запуска
Для обычного запуска используется `cuprof-sass`. У него есть следующие флаги:
- `--output` - директория итогового файла с его названием. Пример - results/test.
- `--arch` - архитектура видеокарты, под которую собран CUDA-бинарник. Например `sm_75`.
- `--sampling-period` - период PC Sampling. Можно выбрать `min`, `low`, `mid`,
  `high` или `max`. По умолчанию используется `low`.
- `--sampling-period-exp` - точный период как степень двойки от 5 до 31.
  Используется вместо `--sampling-period`.
- `--cuprof` - путь к самому профилировщику. По умолчанию используется `build/cuprof`.
- `--flamegraph` - путь к `flamegraph.pl`. Если указать его, вместе с folded
  будет сразу создан SVG. Если не указать, останутся только folded-файлы.
- `-h` или `--help` - показать короткую справку.

После всех флагов ставится `--`, а дальше указывается CUDA-программа и её
аргументы:

```bash
./cuprof-sass [флаги] -- программа [аргументы программы]
```

Сам `cuprof` обычно нужен ручного запуска. У него есть следубщие флаги:
- `--stall-mode` - включить основной режим анализа GPU stalls. Без него
  запускается старый non-stall режим.
- `--sampling-period` - период GPU PC Sampling: `min`, `low`, `mid`, `high` или
  `max`. Чем меньше период, тем чаще берутся сэмплы и тем выше overhead.
- `--sampling-period-exp` - задать период 2^N GPU-циклов, где N от 5 до 31.
  Например, значение `14` означает 16384 цикла.
- `--show-all-stalls` - не отбрасывать `STALL_NONE` и показать вообще все
  причины, которые вернул CUPTI.
- `-f` или `--freq` - частота CPU-сэмплирования в герцах для non-stall режима. Например `-f 99`.
- `-p` или `--profile` - какие CPU-фреймы оставлять в стеке для non-stall режима:
  `minimal`, `standard`, `full` или `debug`. По умолчанию используется `standard`.
- `-h` или `--help` - показать справку.

Профили фильтрации:
- `minimal` - код программы и основные функции CUDA API;
- `standard` - то же самое плюс системные библиотеки. Это обычный вариант;
- `full` - добавляются внутренности CUDA Driver/Runtime и неизвестные функции;
- `debug` - показывается вообще всё, включая внутренности самого профилировщика.

Для CPU `perf_event` нужен `sudo`. Для stall mode `sudo` не нужен.

## Сравнение периодов

Запустить все собранные samples без профилировщика и со всеми пятью периодами:

```bash
scripts/run_sampling_experiment.sh \
  --output results/experiment \
  --arch sm_75 \
  --repeats 5
```

Скрипт проверяет стандартные `min/low/mid/high/max` и промежуточные периоды
`2^6`, `2^7`, `2^8`, `2^9`, `2^10`, `2^12`, `2^14`, `2^16`. Он сохраняет folded, SASS folded,
stderr и SVG отдельно для каждого sample. В `summary.tsv` записываются медиана
времени и замедление, а в `quality.tsv` — фактический период, число PC records,
потери и число folded-строк. Папка для нового эксперимента должна быть пустой,
чтобы результаты разных запусков не смешивались.

## Сборка
```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```

Семплы собираются под архитектуру установленной видеокарты. Проверить, что у
бинарника действительно есть нужный cubin, можно так:

```bash
cuobjdump -lelf build/samples/matrixMul/matrixMul
```

SASS берётся из cubin внутри указанного бинарника. Поэтому архитектура бинарника
должна совпадать с архитектурой GPU. Если точного совпадения функции и PC нет,
скрипт ничего не угадывает и оставляет исходный PC.

## FlameGraph
Для визуализации клонируйте себе репозиторий FlameGraph Брендана Грегга (рядом с проектом профилировщика):

```bash
git clone https://github.com/brendangregg/FlameGraph.git ../FlameGraph
```

## Обычный запуск
Самый простой способ - использовать `cuprof-sass`. Он сам запустит профайлер,
добавит названия SASS-инструкций и сделает итоговый flame graph:

```bash
./cuprof-sass \
  --output results/manual/stall/matrix_mul \
  --sampling-period low \
  --arch sm_75 \
  --flamegraph ../FlameGraph/flamegraph.pl \
  -- build/samples/matrixMul/matrixMul
```

В итоге рядом появятся файлы:

```text
matrix_mul.folded
matrix_mul_sass.folded
matrix_mul.stderr
matrix_mul_sass.stderr
matrix_mul.svg
```

В проекте есть несколько готовых CUDA-примеров:

- `vectorAdd` - простое сложение векторов;
- `matrixMul` - умножение матриц;
- `relu` - простая функция активации;
- `gelu` - функция активации с более сложными вычислениями;
- `softmax` - обработка строк матрицы с reduction и синхронизацией потоков.
- `matrixCompare` - три последовательных варианта умножения матриц: простой,
  с накоплением в регистре и с использованием shared memory.

Все они после сборки находятся в `build/samples/<имя>/<имя>`.

В stderr-файлах лежит статистика профайлера и SASS-корреляции.

Если запустить команду ещё раз с тем же `--output`, старые SASS и SVG сначала
удалятся. Поэтому после ошибки рядом с новым raw-файлом не останется старый граф.

Период сэмплирования можно выбрать такой:

```text
min
low
mid
high
max
```

Если стандартные варианты не подходят, можно указать промежуточный период:

```bash
./cuprof-sass \
  --output results/manual/matrix_16384 \
  --arch sm_75 \
  --sampling-period-exp 14 \
  -- build/samples/matrixMul/matrixMul
```

CUPTI принимает показатель от 5 до 31. Сам период равен `2^N` циклов, поэтому
произвольное число циклов указать нельзя.

Аргументы программы пишутся после её имени:

```bash
./cuprof-sass \
  --output results/manual/my_test \
  --arch sm_75 \
  -- build/my_program arg1 arg2
```

## Что делает cuprof-sass
Внутри всё по-прежнему разделено на несколько простых шагов:

```text
cuprof -> raw folded -> enrich_sass.py -> enriched folded -> flamegraph.pl
```

## Ручной запуск
Сначала снимаем raw stall-профиль:

```bash
cd build
./cuprof --stall-mode --sampling-period low \
  ./samples/matrixMul/matrixMul \
  > ../results/manual/stall/matrix_mul.folded \
  2> ../results/manual/stall/matrix_mul.stderr
cd ..
```

Потом добавляем SASS-инструкции:

```bash
python3 scripts/enrich_sass.py \
  --input results/manual/stall/matrix_mul.folded \
  --binary build/samples/matrixMul/matrixMul \
  --arch sm_75 \
  --output results/manual/stall/matrix_mul_sass.folded \
  2> results/manual/stall/matrix_mul_sass.stderr
```

И собираем итоговый flame graph:

```bash
../FlameGraph/flamegraph.pl \
  results/manual/stall/matrix_mul_sass.folded \
  > results/manual/stall/matrix_mul.svg
```

## Non-stall режим
Он показывает CUDA-кернелы в общем контексте CPU-профиля:

```bash
cd build
sudo ./cuprof -f 99 ./samples/matrixMul/matrixMul > nonstall.folded
```
