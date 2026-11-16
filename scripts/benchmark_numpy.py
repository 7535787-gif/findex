"""Compare the original Python scorer with the NumPy implementation."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from findex.index import DocMeta, Posting, load_index
from findex.index_model import Index
from findex.ranking import BM25, search, search_numpy


def median_ms(call, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("index.bin"))
    parser.add_argument("--out", type=Path, default=Path("docs/numpy-benchmark.md"))
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()

    postings, documents, _ = load_index(args.index)
    index = Index(postings, documents)
    eligible = [
        (term, len(items)) for term, items in index.items() if len(term) >= 3 and items
    ]
    common = max(eligible, key=lambda pair: pair[1])[0]
    rare = min(eligible, key=lambda pair: pair[1])[0]
    queries = [
        ("Частий термін", [common]),
        ("Рідкісний термін", [rare]),
        ("Три терміни", ["людина", "вода", "роки"]),
    ]
    rows = []
    for label, terms in queries:
        if not all(term in index for term in terms):
            continue
        old = median_ms(
            lambda terms=terms: search(index, terms, BM25(), top_k=5),
            args.repeats,
        )
        new = median_ms(
            lambda terms=terms: search_numpy(index, terms, BM25(), top_k=5),
            args.repeats,
        )
        matched_docs = {
            posting.doc_id for term in terms for posting in index.get(term, ())
        }
        rows.append(
            (
                label,
                " ".join(terms),
                len(matched_docs),
                old,
                new,
                old / new if new else 0.0,
            )
        )

    lines = [
        "# NumPy ranking benchmark",
        "",
        f"Index: {len(index.documents)} documents, {len(index)} terms.",
        (
            f"Median of {args.repeats} runs; BM25, top-k=5; "
            "wall time includes snippet generation."
        ),
        "",
        "| Тип запиту | Запит | Зіставлені документи | Python, мс | NumPy, мс | Прискорення |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {label} | `{query}` | {matched} | {old:.3f} | {new:.3f} | {speedup:.2f}× |"
        for label, query, matched, old, new, speedup in rows
    )
    # A controlled scale check makes the vectorized path measurable even though
    # the supplied real corpus contains only eleven documents.
    shared = args.out.parent / "benchmark-shared.txt"
    shared.write_text("common term benchmark", encoding="utf-8")
    scale = 10_000
    scale_docs = {
        doc_id: DocMeta(doc_id, str(shared), f"synthetic-{doc_id}", 100)
        for doc_id in range(scale)
    }
    scale_postings = {
        "common": [Posting(doc_id, 2) for doc_id in range(scale)],
        "term": [Posting(doc_id, 1) for doc_id in range(scale)],
    }
    scale_index = Index(scale_postings, scale_docs)
    old_scale = median_ms(
        lambda: search(scale_index, ["common", "term"], BM25(), top_k=10),
        args.repeats,
    )
    new_scale = median_ms(
        lambda: search_numpy(scale_index, ["common", "term"], BM25(), top_k=10),
        args.repeats,
    )
    scale_speedup = old_scale / new_scale if new_scale else 0.0
    lines += [
        "",
        "## Контрольний тест масштабування",
        "",
        f"Синтетичний індекс із {scale:,} документами та довгими списками постингів:",
        "",
        "| Конфігурація | Python, мс | NumPy, мс | Прискорення |",
        "|---|---:|---:|---:|",
        (
            f"| 10 000 документів, два терміни | {old_scale:.3f} | "
            f"{new_scale:.3f} | {scale_speedup:.2f}× |"
        ),
        "",
        (
            "На реальному корпусі короткі списки постингів не амортизують накладні "
            "витрати NumPy, тому реалізація автоматично залишає Python-шлях для "
            "менш ніж 64 постингів. Для довгих списків вмикається векторизований "
            "шлях з ufunc та argpartition. Контрольний тест показує його поведінку "
            "на навантаженні, де векторизація має сенс."
        ),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
