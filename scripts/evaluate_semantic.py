"""Precision@5 evaluation for keyword, semantic, and hybrid retrieval.

Uses the ten labelled Lab 3 queries plus five paraphrases with manually
assigned relevant document IDs. Run after generating embeddings for the same
index with ``findex embed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from findex.evaluate import RELEVANT, precision_at_k
from findex.index import load_index
from findex.index_model import Index
from findex.ranking import BM25, search
from findex.semantic import DEFAULT_MODEL, hybrid_search, semantic_search
from findex.tokenize import tokenize

PARAPHRASES: dict[str, set[int]] = {
    "drinkable fluid aquatic solvent": {2, 8, 9},
    "lifespan chronology mortality duration": {1, 2, 5, 8, 9},
    "stellar celestial bodies nocturnal observation": {4, 5, 9},
    "motorcar automobile powered transport": {6, 9},
    "humanity hominid species individual": {1, 2, 3, 8, 9},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("index-lab04.bin"))
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings-lab3.npy"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    postings, documents, _ = load_index(args.index)
    index = Index(postings, documents)
    queries = {**RELEVANT, **PARAPHRASES}
    totals = {"keyword": [], "semantic": [], "hybrid": []}

    print("| Запит | Keyword P@5 | Semantic P@5 | Hybrid P@5 |")
    print("|---|---:|---:|---:|")
    for query, relevant in queries.items():
        keyword = search(index, list(tokenize(query)), BM25(), top_k=5)
        semantic = semantic_search(
            index, query, args.embeddings, model_name=args.model, top_k=5
        )
        hybrid = hybrid_search(
            index, query, args.embeddings, model_name=args.model, top_k=5
        )
        scores = {
            "keyword": precision_at_k(keyword, relevant),
            "semantic": precision_at_k(semantic, relevant),
            "hybrid": precision_at_k(hybrid, relevant),
        }
        for mode, score in scores.items():
            totals[mode].append(score)
        print(
            f"| {query} | {scores['keyword']:.2f} | "
            f"{scores['semantic']:.2f} | {scores['hybrid']:.2f} |"
        )

    means = {mode: sum(values) / len(values) for mode, values in totals.items()}
    print(
        f"| **Середнє (15 запитів)** | **{means['keyword']:.2f}** | "
        f"**{means['semantic']:.2f}** | **{means['hybrid']:.2f}** |"
    )
    best = max(means, key=means.get)
    print(
        "\\nІнтерпретація: найвище середнє Precision@5 у режиму "
        f"**{best}** ({means[best]:.2f})."
    )
    print(
        "Порівняння потрібно трактувати з урахуванням невеликої вибірки: "
        "10 початкових запитів і 5 перефразів; якість залежить від ручних "
        "міток релевантності."
    )


if __name__ == "__main__":
    main()
