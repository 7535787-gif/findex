from __future__ import annotations

import numpy as np

from findex.index import DocMeta, Posting
from findex.index_model import Index
from findex.ranking import SearchResult
from findex.semantic import embed_index, reciprocal_rank_fusion, semantic_search


class FakeEncoder:
    def encode(self, texts, **kwargs):
        rows = []
        for text in texts:
            lower = text.lower()
            rows.append(
                [1.0, 0.0, 0.0] if "space" in lower or "cosmos" in lower
                else [0.0, 1.0, 0.0]
            )
        return np.asarray(rows, dtype=np.float32)


def test_embedding_pipeline_and_semantic_search(tmp_path, monkeypatch):
    import findex.semantic as semantic

    monkeypatch.setattr(semantic, "_load_model", lambda model_name: FakeEncoder())
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text("space stars galaxy cosmos", encoding="utf-8")
    two.write_text("water river ocean liquid", encoding="utf-8")
    index = Index(
        {"space": [Posting(1, 1)], "water": [Posting(2, 1)]},
        {
            1: DocMeta(1, str(one), "Astronomy", 4),
            2: DocMeta(2, str(two), "Water", 4),
        },
    )
    path, count = embed_index(index, tmp_path / "vectors.npy", model_name="fake")
    assert count == 2
    assert path.exists()
    matrix = np.load(path)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    results = semantic_search(
        index, "cosmos", path, model_name="fake", top_k=2
    )
    assert results[0].doc_id == 1
    assert results[0].match_type == "semantic"


def test_reciprocal_rank_fusion_combines_rankings():
    lexical = [
        SearchResult(3.0, 1, "one", "one.txt"),
        SearchResult(2.0, 2, "two", "two.txt"),
    ]
    dense = [
        SearchResult(0.9, 2, "two", "two.txt", match_type="semantic"),
        SearchResult(0.8, 3, "three", "three.txt", match_type="semantic"),
    ]
    fused = reciprocal_rank_fusion(lexical, dense, top_k=3)
    assert fused[0].doc_id == 2
    assert {item.doc_id for item in fused} == {1, 2, 3}
    assert all(item.match_type == "hybrid" for item in fused)
