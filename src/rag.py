#!/usr/bin/env python3
"""
RAG System Entrypoint for Corporate Annual Reports.

This script is the main entry point for running RAG experiments.
It orchestrates the modular components from the rag_system package.

Usage Examples:
    # Baseline (No Retrieval)
    python src/rag.py --retriever None --generator gpt-4o-mini
    
    # System 1: OpenAI Semantic + GPT-4o-mini
    python src/rag.py --retriever semantic --embedding_type openai --generator gpt-4o-mini
    
    # System 2: Hybrid (BM25+E5) + GPT-4o-mini
    python src/rag.py --retriever hybrid --generator gpt-4o-mini
    
    # System 3: OpenAI Semantic + Ollama Llama3
    python src/rag.py --retriever semantic --embedding_type openai --generator ollama-llama3.2
    
    # System 4: Hybrid (BM25+E5) + Ollama Llama3
    python src/rag.py --retriever hybrid --generator ollama-llama3.2

Supported Configurations:
    Retrievers:
        - None: Baseline, no retrieval
        - semantic: Vector retrieval (OpenAI or local embeddings)
        - hybrid: BM25 + E5-large with RRF fusion and MMR re-ranking
    
    Generators:
        - gpt-4o-mini: OpenAI GPT-4o-mini via CMU AI Gateway
        - ollama-llama3.2: Llama 3.2 via local Ollama server
        - ollama-llama3: Llama 3 via local Ollama server
"""

import argparse
from pathlib import Path
from typing import List

# Import modular components
from rag_system.data_loader import load_questions, load_chunks
from rag_system.generator_factory import get_generator
from rag_system.retriever_factory import setup_retriever
from rag_system.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG system / No-retrieval baseline for corporate annual reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Baseline (No Retrieval)
    python src/rag.py --retriever None --generator gpt-4o-mini
    
    # RAG with OpenAI embeddings
    python src/rag.py --retriever semantic --embedding_type openai --generator gpt-4o-mini
    
    # RAG with Hybrid (BM25+E5) retriever
    python src/rag.py --retriever hybrid --generator gpt-4o-mini
    
    # RAG with Ollama (local LLM)
    python src/rag.py --retriever semantic --generator ollama-llama3.2
        """
    )
    
    # Retriever selection
    parser.add_argument(
        "--retriever",
        type=str,
        default="None",
        choices=["None", "semantic", "hybrid"],
        help="Retriever type: 'None' (baseline), 'semantic' (vector), or 'hybrid' (BM25+E5)"
    )
    
    # Embedding type (for semantic retriever)
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="openai",
        choices=["openai", "local", "e5-large"],
        help="Embedding type for semantic retriever: 'openai', 'local', or 'e5-large'"
    )
    
    # Generator selection
    parser.add_argument(
        "--generator",
        type=str,
        default="gpt-4o-mini",
        help="LLM generator: 'gpt-4o-mini' (API) or 'ollama-llama3.2' (local)"
    )
    
    # Input/Output paths
    parser.add_argument(
        "--question_file",
        type=str,
        default="data/qa/question.tsv",
        help="Path to input questions TSV"
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="data/corpus",
        help="Directory containing chunk files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/prediction",
        help="Directory to save prediction TSV files"
    )
    
    return parser.parse_args()


def save_predictions(predictions: List[str], args: argparse.Namespace) -> None:
    """Save predictions to output TSV file.
    
    Args:
        predictions: List of prediction strings
        args: Parsed arguments (for output path and naming)
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename: <retriever>_<generator>.tsv
    # For semantic retriever, include embedding type
    if args.retriever == "semantic":
        retriever_name = f"semantic-{args.embedding_type}"
    elif args.retriever == "hybrid":
        retriever_name = "hybrid"
    else:
        retriever_name = "None"
    
    # Sanitize generator name for filename
    generator_name = args.generator.replace("/", "-")
    
    filename = f"{retriever_name}_{generator_name}.tsv"
    out_path = out_dir / filename
    
    with out_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"[INFO] Saved {len(predictions)} predictions to {out_path}")


def main() -> None:
    """Main entry point for RAG system."""
    args = parse_args()
    
    print("=" * 60)
    print("RAG System for Corporate Annual Reports")
    print("=" * 60)
    print(f"[CONFIG] Retriever: {args.retriever}")
    if args.retriever == "semantic":
        print(f"[CONFIG] Embedding Type: {args.embedding_type}")
    print(f"[CONFIG] Generator: {args.generator}")
    print(f"[CONFIG] Questions: {args.question_file}")
    print(f"[CONFIG] Corpus: {args.corpus_dir}")
    print(f"[CONFIG] Output: {args.output_dir}")
    print("=" * 60)
    
    # 1. Load Questions
    questions = load_questions(args.question_file)
    
    # 2. Initialize Generator
    llm = get_generator(args.generator)
    
    # 3. Initialize Retriever (if not baseline)
    retriever = None
    if args.retriever != "None":
        chunks = load_chunks(args.corpus_dir)
        retriever = setup_retriever(
            retriever_type=args.retriever,
            embedding_type=args.embedding_type,
            corpus_dir=args.corpus_dir,
            chunks=chunks
        )
    
    # 4. Run Pipeline
    predictions = run_pipeline(questions, llm, retriever)
    
    # 5. Save Results
    save_predictions(predictions, args)
    
    print("=" * 60)
    print("[DONE] Pipeline execution complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
