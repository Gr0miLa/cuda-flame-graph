# cuda-flame-graph

Это тестовая версия профилировщика для построения флем графов cuda программ. Тут коротко будет описано что в целом сделано и как с этим всем работать.

## Сборка
### Сборка семплов
Для сборки семплов директории проекта выполните:

```
mkdir build && cd build
```

Затем:

```
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j16
```

### Сборка профайлера (если не собран)

По умолчанию профайлер - `pti_loader`. 
Если нужно пересобрать - в директории проекта выполните:
```
g++ -shared -fPIC -o libcupti_prof.so profiler.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcupti -lcuda -lcudart

g++ -o pti_loader loader/loader.cc -I./loader -I./utils -DTOOL_NAME=cupti_prof -lpthread -ldl
```

## Пример запуска

Запустите профайлер на одном из собранных семплов:

```
PROFILER_FREQ=999 ./pti_loader build/matrixMul > output.folded
```

В файле будет весь стек вызовов программы

## Визуализация
Для визуализации клонируйте себе репозиторий __[FlameGraph](https://github.com/brendangregg/FlameGraph)__ Брендана Грегга

```
git clone git@github.com:brendangregg/FlameGraph.git
```

Затем прогоните полученный output.folded через flamegraph.pl 
(пример как запустить из директории проекта):

```
../FlameGraph/./flamegraph.pl output.folded > output.svg
```

Получился итоговый флейм граф `output.svg`