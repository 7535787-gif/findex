"""FastAPI application and HTTP routes."""

from __future__ import annotations

import html
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from findex.index import load_index
from findex.index_model import Index

from .config import Settings
from .dependencies import get_index, get_settings
from .schemas import (
    DocumentResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    StatsResponse,
)
from .service import read_document, run_search

logger = logging.getLogger("findex.web")
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None,
    *,
    search_mode: str | None = None,
    load_on_startup: bool = True,
) -> FastAPI:
    """Create an app; tests can inject settings and an in-memory Index."""

    runtime_settings = settings or Settings()
    mode = search_mode or runtime_settings.search_mode
    if mode not in {"def", "async"}:
        raise ValueError("search_mode must be 'def' or 'async'")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        application.state.started_at = time.perf_counter()
        application.state.index = None
        if load_on_startup and runtime_settings.index_path is not None:
            if not runtime_settings.index_path.is_file():
                raise FileNotFoundError(
                    f"INDEX_PATH does not exist: {runtime_settings.index_path}"
                )
            postings, documents, _ = load_index(runtime_settings.index_path)
            application.state.index = Index(postings, documents)
            logger.info(
                "Search index loaded: documents=%d terms=%d",
                application.state.index.num_docs,
                len(application.state.index),
            )
        yield
        application.state.index = None

    application = FastAPI(
        title="findex — Web Search",
        version="0.7.0",
        description=(
            "HTTP API and web interface for the findex full-text search engine."
        ),
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.index = None
    application.state.started_at = time.perf_counter()
    application.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

    def highlight_snippet(value: str) -> Markup:
        escaped = html.escape(value)
        highlighted = re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", escaped)
        return Markup(highlighted)

    templates.env.filters["highlight"] = highlight_snippet

    @application.middleware("http")
    async def request_context(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request error request_id=%s", request_id)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @application.get("/health", response_model=HealthResponse)
    def health(index: Index = Depends(get_index)) -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/stats", response_model=StatsResponse)
    def stats(
        request: Request,
        index: Index = Depends(get_index),
        settings: Settings = Depends(get_settings),
    ) -> StatsResponse:
        index_path = settings.index_path
        index_size_bytes = (
            index_path.stat().st_size
            if index_path is not None and index_path.is_file()
            else 0
        )

        return StatsResponse(
            status="ok",
            index_size_bytes=index_size_bytes,
            documents=index.num_docs,
            terms=len(index),
            average_document_length=index.avg_doc_length,
            uptime_seconds=max(0.0, time.perf_counter() - request.app.state.started_at),
        )

    def make_search_response(
        request: SearchRequest,
        index: Index,
        settings: Settings,
    ) -> SearchResponse:
        started = time.perf_counter()
        # Fetch all matches before slicing so pagination has a stable total.
        results = run_search(
            index,
            request.q,
            request.scorer,
            index.num_docs,
            mode=request.mode,
            embeddings_path=settings.embeddings_path,
            model_name=settings.embedding_model,
        )
        total = len(results)
        start = (request.page - 1) * request.k
        page_results = results[start : start + request.k]
        return SearchResponse(
            query=request.q,
            mode=request.mode,
            total=total,
            page=request.page,
            pages=(total + request.k - 1) // request.k,
            k=request.k,
            took_ms=(time.perf_counter() - started) * 1000,
            results=[
                SearchResultResponse(
                    doc_id=item.doc_id,
                    title=item.title,
                    score=item.score,
                    snippet=item.snippet,
                    match_type=item.match_type,
                )
                for item in page_results
            ],
        )

    if mode == "def":

        @application.get("/search", response_model=SearchResponse)
        def search_sync(
            q: str = Query(..., min_length=1, max_length=200),
            k: int = Query(10, ge=1, le=100),
            scorer: str = Query("bm25", pattern="^(bm25|tfidf)$"),
            mode: str | None = Query(None, pattern="^(bm25|tfidf|semantic|hybrid)$"),
            page: int = Query(1, ge=1),
            index: Index = Depends(get_index),
            settings: Settings = Depends(get_settings),
        ) -> SearchResponse:
            params = SearchRequest(
                q=q, k=k, scorer=scorer, mode=mode or scorer, page=page
            )
            return make_search_response(params, index, settings)
    else:

        @application.get("/search", response_model=SearchResponse)
        async def search_async(
            q: str = Query(..., min_length=1, max_length=200),
            k: int = Query(10, ge=1, le=100),
            scorer: str = Query("bm25", pattern="^(bm25|tfidf)$"),
            mode: str | None = Query(None, pattern="^(bm25|tfidf|semantic|hybrid)$"),
            page: int = Query(1, ge=1),
            index: Index = Depends(get_index),
            settings: Settings = Depends(get_settings),
        ) -> SearchResponse:
            # Deliberately synchronous CPU work: this mode exists only for M2
            # comparison and demonstrates why async def must not block the loop.
            params = SearchRequest(
                q=q, k=k, scorer=scorer, mode=mode or scorer, page=page
            )
            return make_search_response(params, index, settings)

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def document_api(
        doc_id: int,
        index: Index = Depends(get_index),
    ) -> DocumentResponse:
        document = index.documents.get(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            content = read_document(document)
        except (OSError, ValueError) as exc:
            logger.exception("Cannot read document doc_id=%s", doc_id)
            raise HTTPException(
                status_code=404, detail="Document content unavailable"
            ) from exc
        return DocumentResponse(
            doc_id=doc_id,
            title=document.title,
            path=document.path,
            text=content,
        )

    @application.get(
        "/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False
    )
    def document_page(
        request: Request,
        doc_id: int,
        index: Index = Depends(get_index),
    ) -> HTMLResponse:
        document = index.documents.get(doc_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            content = read_document(document)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=404, detail="Document content unavailable"
            ) from exc
        return templates.TemplateResponse(
            request=request,
            name="document.html",
            context={"document": document, "content": content},
        )

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        q: str = Query("", max_length=200),
        k: int = Query(10, ge=1, le=100),
        scorer: str = Query("bm25", pattern="^(bm25|tfidf)$"),
        mode: str | None = Query(None, pattern="^(bm25|tfidf|semantic|hybrid)$"),
        page: int = Query(1, ge=1),
        index: Index | None = Depends(optional_index),
        settings: Settings = Depends(get_settings),
    ) -> HTMLResponse:
        results = None
        error = None
        if q and index is not None:
            try:
                results = make_search_response(
                    SearchRequest(
                        q=q, k=k, scorer=scorer, mode=mode or scorer, page=page
                    ),
                    index,
                    settings,
                )
            except Exception:
                logger.exception(
                    "Web UI search failed request_id=%s", request.state.request_id
                )
                error = "Не вдалося виконати пошук. Спробуйте ще раз."
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "query": q,
                "k": k,
                "scorer": scorer,
                "mode": mode or scorer,
                "page": page,
                "response": results,
                "error": error,
                "index_loaded": index is not None,
                "app_name": settings.app_name,
            },
        )

    return application


def optional_index(request: Request) -> Index | None:
    return getattr(request.app.state, "index", None)


app = create_app()
