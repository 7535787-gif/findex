from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import DocMeta, Posting
from findex.index_model import Index
from findex.web.app import create_app, optional_index
from findex.web.config import Settings
from findex.web.dependencies import get_index


@pytest.fixture
def tiny_index(tmp_path: Path) -> Index:
    first = tmp_path / "python.txt"
    second = tmp_path / "asyncio.txt"
    first.write_text(
        "Python makes search useful. Python supports tools.", encoding="utf-8"
    )
    second.write_text(
        "Asyncio supports concurrent network applications.", encoding="utf-8"
    )
    documents = {
        1: DocMeta(1, str(first), "Python guide", 8),
        2: DocMeta(2, str(second), "Asyncio guide", 6),
    }
    postings = {
        "python": [Posting(1, 2)],
        "search": [Posting(1, 1)],
        "asyncio": [Posting(2, 1)],
        "concurrent": [Posting(2, 1)],
    }
    return Index(postings, documents)


@pytest.fixture
def client(tiny_index: Index):
    application = create_app(
        Settings(index_path=None),
        load_on_startup=False,
    )
    application.dependency_overrides[get_index] = lambda: tiny_index
    application.dependency_overrides[optional_index] = lambda: tiny_index
    with TestClient(application) as test_client:
        yield test_client
    application.dependency_overrides.clear()


def test_search_success_and_pagination(client: TestClient) -> None:
    response = client.get("/search", params={"q": "python", "k": 1, "page": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "python"
    assert payload["total"] == 1
    assert payload["results"][0]["title"] == "Python guide"
    assert payload["results"][0]["doc_id"] == 1


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "python", "k": 0},
        {"q": "python", "k": 101},
        {"q": "python", "page": 0},
        {"q": "python", "scorer": "unknown"},
        {"q": "x" * 201},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, object]) -> None:
    response = client.get("/search", params=params)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_unknown_document_is_404(client: TestClient) -> None:
    response = client.get("/docs/999")
    assert response.status_code == 404


def test_document_api_returns_content(client: TestClient) -> None:
    response = client.get("/docs/1")
    assert response.status_code == 200
    assert "Python makes search useful" in response.json()["text"]


def test_health_is_503_when_index_is_not_loaded() -> None:
    application = create_app(Settings(index_path=None), load_on_startup=False)
    with TestClient(application) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 503


def test_stats_and_ui(client: TestClient) -> None:
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["documents"] == 2
    page = client.get("/", params={"q": "python"})
    assert page.status_code == 200
    assert "Python guide" in page.text
    assert "<mark>Python</mark>" in page.text


def test_openapi_available(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/search" in response.json()["paths"]


def test_request_id_is_returned(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "test-123"})
    assert response.headers["x-request-id"] == "test-123"


def test_twenty_concurrent_search_requests(tiny_index: Index) -> None:
    application = create_app(Settings(index_path=None), load_on_startup=False)
    application.dependency_overrides[get_index] = lambda: tiny_index
    application.dependency_overrides[optional_index] = lambda: tiny_index
    with (
        TestClient(application) as test_client,
        ThreadPoolExecutor(max_workers=20) as pool,
    ):
            responses = list(
                pool.map(
                    lambda _: test_client.get("/search", params={"q": "python"}),
                    range(20),
                )
            )
    assert all(response.status_code == 200 for response in responses)


def test_unexpected_exception_returns_clean_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    app_module = importlib.import_module("findex.web.app")

    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app_module, "run_search", explode)
    response = client.get("/search", params={"q": "python"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "secret internal detail" not in response.text
    assert response.headers.get("x-request-id")


def test_lifespan_loads_index_once(tmp_path: Path) -> None:
    from findex.index import save_index

    source = tmp_path / "source.txt"
    source.write_text("python search", encoding="utf-8")
    postings = {"python": [Posting(7, 1)]}
    documents = {7: DocMeta(7, str(source), "Loaded index", 2)}
    index_path = tmp_path / "index.bin"
    save_index(postings, documents, index_path)

    application = create_app(Settings(index_path=index_path))
    with TestClient(application) as test_client:
        assert test_client.get("/health").status_code == 200
        stats = test_client.get("/stats").json()
        assert stats["documents"] == 1
        assert test_client.get("/search", params={"q": "python"}).json()["total"] == 1


def test_document_html_page(client: TestClient) -> None:
    response = client.get("/doc/1")
    assert response.status_code == 200
    assert "Python guide" in response.text
    assert "Python makes search useful" in response.text
