# backend/core/__init__.py
# exposes get_retriever so other modules import cleanly

from .storage import get_retriever
__all__ = ["get_retriever"]