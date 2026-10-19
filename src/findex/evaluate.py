
from pathlib import Path

from .index_model import open_index
from .ranking import BM25, TfIdf, search
from .tokenize import tokenize

# ID відповідають doc_id з індексу:
# doc_id=0 — перший документ, data/0.txt;
# doc_id=1 — другий документ, data/1.txt тощо.
# Релевантність визначається за змістом документа для конкретного запиту.
RELEVANT = {
    "людина": {1, 2, 3, 8, 9},
    "вода": {2, 8, 9},
    "роки": {1, 2, 5, 8, 9},
    "життя": {1, 2, 6, 7, 9},
    "земля": {2},
    "дикун": {0, 2},
    "кози": {0},
    "океан": {0},
    "астроном": {4, 5, 9},
    "машина": {6, 9},
    
}

QUERIES = list(RELEVANT)


def precision_at_k(results, relevant_ids, k=5):
    top_results = results[:k]
    if len(top_results) < k:
        return len(
            [result for result in top_results if result.doc_id in relevant_ids]
        ) / k
    return sum(
        result.doc_id in relevant_ids for result in top_results
    ) / k


def main():
    with open_index(Path("index.bin")) as index:
        totals = {"TfIdf": [], "BM25": []}

        print("Manual relevance labels:", RELEVANT)

        for query in QUERIES:
            terms = list(tokenize(query))
            print(f"\nQuery: {query}")

            for scorer in (TfIdf(), BM25()):
                name = type(scorer).__name__
                results = search(index, terms, scorer, top_k=5)
                p_at_5 = precision_at_k(results, RELEVANT[query])

                totals[name].append(p_at_5)

                print(f"\n{name} (P@5={p_at_5:.2f}):")
                for result in results:
                    marker = (
                        "relevant"
                        if result.doc_id in RELEVANT[query]
                        else "not labelled relevant"
                    )
                    print(
                        f"  {result.score:.4f} | doc={result.doc_id} | "
                        f"{result.title} | {marker}"
                    )
                    print(f"    {result.snippet}")

        print("\nPrecision@5 summary")
        print("| Query | TF-IDF | BM25 |")
        print("|---|---:|---:|")

        for query, tfidf_score, bm25_score in zip(
            QUERIES, totals["TfIdf"], totals["BM25"]
        ):
            print(f"| {query} | {tfidf_score:.2f} | {bm25_score:.2f} |")

        print(
            "| **Mean** | "
            f"**{sum(totals['TfIdf']) / len(QUERIES):.2f}** | "
            f"**{sum(totals['BM25']) / len(QUERIES):.2f}** |"
        )


if __name__ == "__main__":
    main()
