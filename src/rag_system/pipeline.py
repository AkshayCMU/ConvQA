"""
Pipeline Module for RAG System.

Contains the core run_pipeline orchestration logic that processes questions
through the retrieval and generation stages.
"""

from typing import List, Dict, Any, Optional

try:
    from langchain.schema.retriever import BaseRetriever
except ImportError:
    from langchain_core.retrievers import BaseRetriever


# =============================================================================
# Prompt Templates
# =============================================================================

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering investor questions about a corporate annual report.

Use the following context from the annual report to answer the question. 
- If the answer can be found in the context, provide a clear, concise answer.
- If the answer is partially in the context, provide what you can and note any limitations.
- If the answer is NOT in the context, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

BASELINE_PROMPT_TEMPLATE = """You are a helpful assistant answering investor questions about corporate annual reports.

Answer the following question based on your general knowledge about corporate finance and business.
If you don't know the specific answer, say "I don't know."

Question: {question}

Answer:"""


def run_pipeline(
    questions: List[Dict[str, str]],
    llm: Any,
    retriever: Optional[BaseRetriever] = None
) -> List[str]:
    """Run the RAG pipeline on a list of questions.
    
    Args:
        questions: List of question dicts with 'text' and 'type' keys
        llm: Initialized LLM generator (ChatOpenAI, ChatOllama, etc.)
        retriever: Optional retriever. If None, runs in baseline (no-retrieval) mode.
        
    Returns:
        List of prediction strings, one per question
    """
    predictions = []
    total = len(questions)
    
    mode = "RAG" if retriever else "Baseline (No Retrieval)"
    print(f"[INFO] Running {mode} pipeline on {total} questions...")
    
    for i, q in enumerate(questions):
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  Processing {i + 1}/{total}...")
        
        try:
            prediction = _process_single_question(q, llm, retriever)
            predictions.append(prediction)
        except Exception as e:
            print(f"[ERROR] Question {i + 1}: {e}")
            predictions.append("Error: Failed to generate response")
    
    print(f"[INFO] Pipeline complete. Generated {len(predictions)} predictions.")
    return predictions


def _process_single_question(
    question: Dict[str, str],
    llm: Any,
    retriever: Optional[BaseRetriever]
) -> str:
    """Process a single question through retrieval and generation.
    
    Args:
        question: Question dict with 'text' and 'type' keys
        llm: LLM generator
        retriever: Optional retriever
        
    Returns:
        Generated answer string
    """
    question_text = question["text"]
    
    if retriever:
        # RAG Mode: Retrieve context and generate
        context = _retrieve_context(question_text, retriever)
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question_text
        )
    else:
        # Baseline Mode: Generate without context
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question_text)
    
    # Generate response
    response = llm.invoke(prompt)
    
    # Extract content from response (handles different LLM response types)
    answer = _extract_response_content(response)
    
    # Clean for TSV output (remove tabs and newlines)
    clean_answer = answer.strip().replace("\n", " ").replace("\t", " ")
    
    return clean_answer


def _retrieve_context(query: str, retriever: BaseRetriever) -> str:
    """Retrieve relevant context for a query.
    
    Args:
        query: Question text
        retriever: Retriever instance
        
    Returns:
        Concatenated context string from retrieved documents
    """
    try:
        docs = retriever.invoke(query)
        
        if not docs:
            return "[No relevant context found]"
        
        # Concatenate document contents with separators
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"Document {i}")
            context_parts.append(f"--- Source: {source} ---\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        print(f"[WARN] Retrieval failed: {e}")
        return "[Retrieval error]"


def _extract_response_content(response: Any) -> str:
    """Extract text content from LLM response.
    
    Handles different response types from various LLM providers.
    
    Args:
        response: LLM response object
        
    Returns:
        Extracted text content
    """
    # Handle LangChain message objects
    if hasattr(response, "content"):
        return response.content
    
    # Handle raw string responses
    if isinstance(response, str):
        return response
    
    # Handle dict responses
    if isinstance(response, dict) and "content" in response:
        return response["content"]
    
    # Fallback: convert to string
    return str(response)
