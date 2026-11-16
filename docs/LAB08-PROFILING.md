# Lab 08 — профілювання та вимірювання

## cProfile

Профілі `index` і `search` створюються командою:

```powershell
uv run python scripts/profile_lab8.py --corpus data --index index.bin
```

Скрипт записує `.prof` і текстові звіти в `docs/profiling/`. Пошуковий запит
`його вода роки` повторюється 100 разів, щоб короткі виклики були помітні в
профілі.

## py-spy

Команда, яка має створювати sampling flame graph:

```powershell
py-spy record -o docs/flamegraph.svg -- .\.venv\Scripts\python.exe scripts/profile_lab8.py --index index.bin
```

У поточному Windows-середовищі `py-spy 0.4.3` тричі завершився помилкою
`Failed to find python version from target process` (з `uv`, з прямим запуском
Python і через executable в `.venv`). Тому `docs/flamegraph.svg` — це
cProfile cumulative-call-tree visualization, а **не** py-spy sampling flame
graph. Не слід представляти його як виконання вимоги py-spy. Для повного
закриття пункту потрібно повторити sampling на сумісному середовищі
(наприклад, Linux) і замінити SVG справжнім результатом py-spy.

## Scalene

Побудова індексу:

```powershell
scalene run --cpu-only --profile-all --outfile docs/profiling/scalene-index.txt scripts/profile_lab8.py --index index.bin
scalene view --cli docs/profiling/scalene-index.json
```

Пошук:

```powershell
scalene run --cpu-only --profile-all --outfile docs/profiling/scalene-search.txt scripts/profile_lab8.py --skip-build --index index.bin
scalene view --cli docs/profiling/scalene-search.json
```

У Scalene розрізняються Python time та native time. У наданому профілі побудови
індексу `tokenize` показав 0% Python і близько 8% native; рядок отримання
`match.group()` також відображений як native. Для пошукового профілю основна
гаряча точка — `make_snippet`, зокрема регулярний вираз і читання фрагментів.

## pytest-benchmark та CI

`pytest-benchmark` входить до dev-залежностей у `pyproject.toml`.
`tests/test_benchmarks.py` містить вимірювання базового `search` та
`build_index` на фікстурному корпусі. Workflow `.github/workflows/ci.yml`
виконує `uv sync --dev`, а потім `uv run pytest -q`, тому benchmark-тести
входять до CI-запуску pytest.

Локальний запуск benchmark-тестів:

```powershell
uv run pytest tests/test_benchmarks.py --benchmark-verbose
```

## Semantic та hybrid retrieval

Після встановлення optional dependency:

```powershell
uv sync --extra semantic
uv run findex embed index-lab04.bin --out embeddings-lab3.npy
uv run python scripts/evaluate_semantic.py --index index-lab04.bin --embeddings embeddings-lab3.npy
```

Оцінювання формує Precision@5 для keyword, semantic та hybrid режимів на
10 початкових запитах і 5 перефразах. Потрібно запускати його в середовищі,
де модель `sentence-transformers` доступна; самі числові результати не
підміняються вигаданими значеннями.
