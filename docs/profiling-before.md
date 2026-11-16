# Profiling results before optimization

Measurements were taken against the supplied full corpus (11 text files, including the large encyclopedia export) and the existing full index (967,436 terms). The index build was profiled once; the three-term search `його вода роки` was profiled over 100 repeated calls to make the short request measurable.

## `findex index` — cProfile

- Total profiled build: **23.52 s** (instrumented run).
- `build_partial`: **21.34 s cumulative / 8.49 s self**. Python document loop, token counting and posting construction.
- `tokenize`: **8.16 s cumulative / 3.05 s self**. Python generator and regex/token handling.
- `unicodedata.normalize`: **2.87 s**. CPython built-in/native Unicode normalization.
- Additional native-heavy work: `_collections._count_elements` **1.26 s**; file reads **1.26 s**.

## `findex search` — cProfile

100 repeated runs of `його вода роки` took **0.049 s** in total (~0.49 ms per call under profiler).

The largest cumulative costs were:
1. `make_snippet`: **0.035 s** total across 1,100 calls; Python snippet preparation and matching.
2. File open: **0.012 s** total across 1,100 calls; operating-system I/O boundary.
3. `tokenize`: **0.008 s** total across 7,200 calls; Python token normalization/regex.

Scorer calls themselves were small on this corpus because the index has only 11 documents and the query lists are short.

## Python versus native work

cProfile identifies Python functions and CPython built-ins but does not provide Scalene's exact Python/native split. Based on the profiled call sites, `build_partial`, `tokenize`, `make_snippet`, and ranking loops are Python work; `unicodedata.normalize`, `_collections._count_elements`, regex engine calls, and file reads execute in native/runtime code. Run the Scalene command in `LAB08-PROFILING.md` on the development machine to obtain the actual percentage split. Scalene and py-spy were not installed in the execution environment used to prepare this patch, so no fabricated sampling percentages are reported.

## Flame view

`docs/flamegraph.svg` is a cProfile cumulative-call-tree view generated from the measured index-build profile. It is explicitly not a py-spy sample flame graph. Use the py-spy command in `LAB08-PROFILING.md` to generate the true sampling flame graph on the workstation and replace this file.
