"""
Data Loading Module for RAG System.

Handles loading questions from TSV files and document chunks from corpus directory.
"""

from pathlib import Path
from typing import List, Dict, Any

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document


def load_questions(question_file: str) -> List[Dict[str, str]]:
    """Load questions from TSV file.
    
    Args:
        question_file: Path to questions TSV file (question\ttype format)
        
    Returns:
        List of dicts with 'text' and 'type' keys
    """
    questions = []
    path = Path(question_file)
    
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
        
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                questions.append({
                    "text": parts[0],
                    "type": parts[1]
                })
            elif len(parts) == 1:
                # Handle lines without type
                questions.append({
                    "text": parts[0],
                    "type": "unknown"
                })
                
    print(f"[INFO] Loaded {len(questions)} questions from {question_file}")
    return questions


def load_chunks(corpus_dir: str) -> List[Document]:
    """Load chunk files as LangChain Documents.
    
    Args:
        corpus_dir: Path to directory containing chunk .txt files
        
    Returns:
        List of LangChain Document objects with page_content and metadata
    """
    chunks = []
    path = Path(corpus_dir)
    
    if not path.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}")
        return []
        
    files = sorted(list(path.glob("*.txt")))
    print(f"[INFO] Found {len(files)} chunk files in {corpus_dir}")
    
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            # Add metadata from filename for traceability
            chunks.append(Document(
                page_content=text, 
                metadata={"source": fpath.name, "path": str(fpath)}
            ))
        except Exception as e:
            print(f"[WARN] Failed to read {fpath.name}: {e}")
            
    print(f"[INFO] Loaded {len(chunks)} document chunks")
    return chunks
