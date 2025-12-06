"""
Retriever Factory Module for RAG System.

Creates and configures retrievers for both API-based and open-weight approaches.

Supported Retrievers:
- semantic: Pure vector retrieval using OpenAI or local embeddings
- hybrid: BM25 + E5-large with RRF fusion and MMR re-ranking

The hybrid retriever implements a two-stage pipeline:
1. Stage 1 (RRF): Combines BM25 (sparse) and E5 (dense) using Reciprocal Rank Fusion
2. Stage 2 (MMR): Re-ranks using Maximal Marginal Relevance for diversity
"""

import os
from pathlib import Path
from typing import List, Any, Optional

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from langchain.schema.retriever import BaseRetriever
except ImportError:
    from langchain_core.retrievers import BaseRetriever


def get_embeddings(embedding_type: str) -> Any:
    """Create and return an embedding model based on the specified type.
    
    Args:
        embedding_type: Type of embeddings. Supported values:
            - "openai": OpenAI text-embedding-3-small via CMU Gateway
            - "local": SentenceTransformers all-MiniLM-L6-v2
            - "e5-large": SentenceTransformers intfloat/e5-large-v2
            
    Returns:
        Embedding model instance
        
    Raises:
        ValueError: If embedding_type is not recognized
    """
    
    if embedding_type == "openai":
        return _create_openai_embeddings()
    elif embedding_type == "local":
        return _create_local_embeddings("all-MiniLM-L6-v2")
    elif embedding_type == "e5-large":
        return _create_local_embeddings("intfloat/e5-large-v2")
    else:
        raise ValueError(
            f"Unknown embedding type: {embedding_type}. "
            "Supported: 'openai', 'local', 'e5-large'"
        )


def _create_openai_embeddings() -> Any:
    """Create OpenAI embeddings via CMU AI Gateway."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. Run: pip install langchain-openai"
        )
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    
    print("[INFO] Initializing OpenAI Embeddings (text-embedding-3-small)")
    
    return OpenAIEmbeddings(
        model="azure/text-embedding-3-small",
        api_key=api_key,
        base_url="https://ai-gateway.andrew.cmu.edu/",
    )


def _create_local_embeddings(model_name: str) -> Any:
    """Create local SentenceTransformer embeddings.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        SentenceTransformerEmbeddings instance
    """
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
    except ImportError:
        raise ImportError(
            "langchain-community not installed. Run: pip install langchain-community sentence-transformers"
        )
    
    print(f"[INFO] Initializing Local Embeddings: {model_name}")
    
    return SentenceTransformerEmbeddings(model_name=model_name)


def setup_retriever(
    retriever_type: str,
    embedding_type: str,
    corpus_dir: str,
    chunks: List[Document]
) -> BaseRetriever:
    """Setup and return a retriever based on the specified type.
    
    Args:
        retriever_type: Type of retriever. Supported values:
            - "semantic": Pure vector retrieval
            - "hybrid": BM25 + E5 with RRF + MMR
        embedding_type: Embedding type for semantic retriever
        corpus_dir: Path to corpus directory (for naming vector store)
        chunks: List of Document objects to index
        
    Returns:
        Configured retriever instance
        
    Raises:
        ValueError: If retriever_type is not recognized or chunks are empty
    """
    
    if not chunks:
        raise ValueError("No chunks provided for retrieval indexing!")
    
    if retriever_type == "semantic":
        return _setup_semantic_retriever(embedding_type, corpus_dir, chunks)
    elif retriever_type == "hybrid":
        return _setup_hybrid_retriever(chunks)
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. "
            "Supported: 'semantic', 'hybrid'"
        )


def _setup_semantic_retriever(
    embedding_type: str,
    corpus_dir: str,
    chunks: List[Document]
) -> BaseRetriever:
    """Setup pure semantic (vector) retriever using ChromaDB.
    
    Args:
        embedding_type: Type of embeddings to use
        corpus_dir: Corpus directory path (used for persist_dir naming)
        chunks: Documents to index
        
    Returns:
        Chroma vector store as retriever
    """
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        raise ImportError(
            "chromadb not installed. Run: pip install chromadb"
        )
    
    print(f"[INFO] Setting up Semantic Retriever (Embeddings: {embedding_type})...")
    
    embeddings = get_embeddings(embedding_type)
    persist_dir = f"data/chroma_db_{embedding_type}"
    
    # Check if vector store already exists
    if Path(persist_dir).exists() and len(list(Path(persist_dir).iterdir())) > 0:
        print(f"[INFO] Loading existing vector store from {persist_dir}")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print(f"[INFO] Creating new vector store in {persist_dir}")
        print(f"[INFO] Indexing {len(chunks)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    print(f"[INFO] Semantic Retriever ready (k=3)")
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def _setup_hybrid_retriever(chunks: List[Document]) -> BaseRetriever:
    """Setup hybrid retriever with BM25 + E5, RRF fusion, and MMR re-ranking.
    
    This implements a two-stage retrieval pipeline:
    1. Stage 1 (RRF): Combines BM25 sparse retrieval with E5 dense retrieval
       using Reciprocal Rank Fusion to merge ranked lists
    2. Stage 2 (MMR): Re-ranks the fused results using Maximal Marginal
       Relevance to balance relevance and diversity
    
    Args:
        chunks: Documents to index
        
    Returns:
        ContextualCompressionRetriever with hybrid pipeline
    """
    # Import Chroma
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma
    
    # Import BM25Retriever
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        from langchain.retrievers import BM25Retriever
    
    # Import EnsembleRetriever
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain_community.retrievers import EnsembleRetriever
        except ImportError:
            from langchain.retrievers.ensemble import EnsembleRetriever
    
    # Import ContextualCompressionRetriever
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    
    # Import EmbeddingsFilter
    try:
        from langchain.retrievers.document_compressors import EmbeddingsFilter
    except ImportError:
        try:
            from langchain_community.document_compressors import EmbeddingsFilter
        except ImportError:
            from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
    
    print("[INFO] Setting up Hybrid (BM25 + E5) Retriever with RRF + MMR...")
    
    # =========================================================================
    # Stage 1a: BM25 Sparse Retriever
    # =========================================================================
    print("[INFO] Initializing BM25 (sparse) retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5  # Retrieve top 5 from BM25
    
    # =========================================================================
    # Stage 1b: E5-Large Dense Retriever
    # =========================================================================
    print("[INFO] Initializing E5-large (dense) retriever...")
    e5_embeddings = get_embeddings("e5-large")
    
    persist_dir_e5 = "data/chroma_db_e5-large"
    
    if Path(persist_dir_e5).exists() and len(list(Path(persist_dir_e5).iterdir())) > 0:
        print(f"[INFO] Loading existing E5 vector store from {persist_dir_e5}")
        vectorstore_e5 = Chroma(
            persist_directory=persist_dir_e5,
            embedding_function=e5_embeddings
        )
    else:
        print(f"[INFO] Creating E5 vector store in {persist_dir_e5}")
        print(f"[INFO] Indexing {len(chunks)} chunks with E5...")
        vectorstore_e5 = Chroma.from_documents(
            documents=chunks,
            embedding=e5_embeddings,
            persist_directory=persist_dir_e5
        )
    
    e5_retriever = vectorstore_e5.as_retriever(search_kwargs={"k": 5})
    
    # =========================================================================
    # Stage 2: RRF Fusion via EnsembleRetriever
    # =========================================================================
    print("[INFO] Creating RRF Ensemble (BM25 + E5)...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, e5_retriever],
        weights=[0.5, 0.5]  # Equal weight; EnsembleRetriever uses RRF by default
    )
    
    # =========================================================================
    # Stage 3: MMR Re-ranking via EmbeddingsFilter
    # =========================================================================
    print("[INFO] Adding MMR re-ranking layer...")
    compressor = EmbeddingsFilter(
        embeddings=e5_embeddings,
        similarity_threshold=0.5,  # Filter out low-relevance docs
        k=3  # Final number of documents to return
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    
    print("[INFO] Hybrid (RRF + MMR) Retriever ready")
    return compression_retriever
