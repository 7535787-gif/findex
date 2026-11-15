"""FastAPI dependencies."""

from fastapi import HTTPException, Request

from findex.index_model import Index

from .config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_index(request: Request) -> Index:
    index = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return index
