
from pathlib import Path

from findex.index_model import open_index
from findex.ranking import BM25, TfIdf, search


def main():
    with open_index(Path("index.bin")) as index:
        bm25 = BM25()
        tfidf = TfIdf()

        print("\n1. Rare term vs common term")

        rare_terms = sorted(
            term for term in index if index.df(term) == 1
        )
        common_terms = sorted(
            (term for term in index if index.df(term) > 1),
            key=lambda term: index.df(term),
            reverse=True,
        )

        found_pair = False

        for rare_term in rare_terms:
            rare_posting = index[rare_term][0]

            for common_term in common_terms:
                common_doc_ids = {
                    posting.doc_id for posting in index[common_term]
                }

                if rare_posting.doc_id in common_doc_ids:
                    continue

                rare_results = search(
                    index, [rare_term], tfidf, top_k=index.num_docs
                )
                common_results = search(
                    index, [common_term], tfidf, top_k=index.num_docs
                )

                rare_rank = next(
                    (
                        rank
                        for rank, result in enumerate(rare_results, 1)
                        if result.doc_id == rare_posting.doc_id
                    ),
                    None,
                )

                common_posting = index[common_term][0]
                common_rank = next(
                    (
                        rank
                        for rank, result in enumerate(common_results, 1)
                        if result.doc_id == common_posting.doc_id
                    ),
                    None,
                )

                print(
                    f"Rare term: {rare_term}, df={index.df(rare_term)}"
                )
                print(
                    f"Common term: {common_term}, "
                    f"df={index.df(common_term)}"
                )
                print(
                    f"Rare-term document id={rare_posting.doc_id}, "
                    f"rank={rare_rank}"
                )
                print(
                    f"Common-term document id={common_posting.doc_id}, "
                    f"rank={common_rank}"
                )
                print(
                    "Rare-term document ranks higher:",
                    rare_rank is not None
                    and common_rank is not None
                    and rare_rank < common_rank,
                )

                found_pair = True
                break

            if found_pair:
                break

        if not found_pair:
            print("Could not find a suitable document pair.")

        print("\n2. BM25 term-frequency saturation")

        term = "людина"
        if index.get(term):
            posting = index[term][0]
            score_one = bm25.score(term, 1, index, posting.doc_id)
            score_twenty = bm25.score(term, 20, index, posting.doc_id)

            print(f"Term: {term}")
            print(f"Score with tf=1: {score_one:.4f}")
            print(f"Score with tf=20: {score_twenty:.4f}")
            print(
                "Saturation demonstrated:",
                score_twenty > score_one
                and score_twenty < score_one * 20,
            )
        else:
            print(f"Term '{term}' not found.")

        print("\n3. Short vs long document")

        candidates = []

        for candidate_term in index:
            postings = [
                posting
                for posting in index[candidate_term]
                if posting.tf == 1
            ]

            for short in postings:
                for long in postings:
                    short_length = index.documents[short.doc_id].length
                    long_length = index.documents[long.doc_id].length

                    if short_length < long_length:
                        candidates.append(
                            (
                                long_length - short_length,
                                candidate_term,
                                short,
                                long,
                            )
                        )

        if candidates:
            _, candidate_term, short, long = max(
                candidates, key=lambda item: item[0]
            )

            short_score = bm25.score(
                candidate_term, 1, index, short.doc_id
            )
            long_score = bm25.score(
                candidate_term, 1, index, long.doc_id
            )

            print(f"Term: {candidate_term}")
            print(
                f"Short doc: id={short.doc_id}, "
                f"length={index.documents[short.doc_id].length}, "
                f"score={short_score:.4f}"
            )
            print(
                f"Long doc: id={long.doc_id}, "
                f"length={index.documents[long.doc_id].length}, "
                f"score={long_score:.4f}"
            )
            print(
                "Short document ranks higher:",
                short_score > long_score,
            )
        else:
            print("No suitable pair found.")


if __name__ == "__main__":
    main()
