"""Retrieval package: local RAG pipeline with a swappable vector backend."""

from arthur.retrieval.chunking import chunk_text
from arthur.retrieval.citations import extract_citations, sanitize_citations
from arthur.retrieval.embeddings import Embedder, EmbeddingError, HashEmbedder, OllamaEmbedder
from arthur.retrieval.service import IndexReport, Passage, Retriever
from arthur.retrieval.store import VectorItem, VectorStore, build_vector_store

__all__ = [
    "Embedder",
    "EmbeddingError",
    "HashEmbedder",
    "IndexReport",
    "OllamaEmbedder",
    "Passage",
    "Retriever",
    "VectorItem",
    "VectorStore",
    "build_vector_store",
    "chunk_text",
    "extract_citations",
    "sanitize_citations",
]
