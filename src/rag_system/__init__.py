"""
rag_system package for modular RAG pipeline.

Modules:
- data_loader: Load questions and document chunks
- generator_factory: Create LLM generators (API and open-weight)
- retriever_factory: Create retrievers (Semantic, Hybrid BM25+E5)
- pipeline: Core RAG pipeline orchestration
"""

from .data_loader import load_questions, load_chunks
from .generator_factory import get_generator
from .retriever_factory import setup_retriever, get_embeddings
from .pipeline import run_pipeline

__all__ = [
    "load_questions",
    "load_chunks",
    "get_generator",
    "setup_retriever",
    "get_embeddings",
    "run_pipeline",
]
