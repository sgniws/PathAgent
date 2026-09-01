"""Model-agnostic pathology image-text retrievers."""

from .base import PathologyRetriever, RetrieverSpec
from .factory import create_retriever
from .config import resolve_retriever_args

__all__ = [
    "PathologyRetriever",
    "RetrieverSpec",
    "create_retriever",
    "resolve_retriever_args",
]
