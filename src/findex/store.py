import json
import time
from pathlib import Path

from .index import DocMeta, Posting


def save_json(index, documents, path: Path) -> float:
    start = time.perf_counter()

    data = {
        "index": {
            term: [
                [posting.doc_id, posting.tf]
                for posting in postings
            ]
            for term, postings in index.items()
        },
        "documents": {
            str(doc_id): [
                document.path,
                document.title,
                document.length,
            ]
            for doc_id, document in documents.items()
        },
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
        )

    return time.perf_counter() - start


def load_json(path):
    json_path = Path(path)
    start = time.perf_counter()
    
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    index = {
        term: [Posting(doc_id, tf) for doc_id, tf in postings]
        for term, postings in data["index"].items()
    }

    documents = {
        int(doc_id): DocMeta(
            doc_id=int(doc_id),
            path=document_path,
            title=title,
            length=length,
        )
        for doc_id, (document_path, title, length)
        in data["documents"].items()
    }

    elapsed = time.perf_counter() - start
    return index, documents, elapsed