"""FastAPI application for the findex search engine."""

from .app import app, create_app
from .config import Settings

__all__ = ["app", "create_app", "Settings"]
