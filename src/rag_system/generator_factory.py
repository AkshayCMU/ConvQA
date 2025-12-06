"""
Generator Factory Module for RAG System.

Creates and configures LLM generators for both API-based and open-weight models.

Supported Generators:
- API-based: gpt-4o-mini (via CMU AI Gateway)
- Open-weight: ollama-llama3 (via local Ollama server)
"""

import os
from typing import Any


def get_generator(generator_name: str, **kwargs) -> Any:
    """Create and return an LLM generator based on the specified name.
    
    Args:
        generator_name: Name of the generator. Supported values:
            - "gpt-4o-mini" or "gpt-4o-mini-2024-07-18": OpenAI GPT-4o-mini via CMU Gateway
            - "ollama-llama3" or "ollama-llama3.2": Llama 3 via local Ollama
        **kwargs: Additional arguments passed to the LLM constructor
        
    Returns:
        Initialized LLM object (ChatOpenAI or ChatOllama)
        
    Raises:
        RuntimeError: If required environment variables are missing
        ValueError: If generator_name is not recognized
    """
    
    # Normalize generator name
    gen_lower = generator_name.lower()
    
    # =========================================================================
    # API-Based Generators (via CMU AI Gateway)
    # =========================================================================
    if gen_lower.startswith("gpt-") or gen_lower.startswith("azure/"):
        return _create_openai_generator(generator_name, **kwargs)
    
    # =========================================================================
    # Open-Weight Generators (via Ollama)
    # =========================================================================
    elif gen_lower.startswith("ollama-"):
        # Extract model name after "ollama-"
        model_name = generator_name.split("-", 1)[1] if "-" in generator_name else "llama3.2"
        return _create_ollama_generator(model_name, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown generator: {generator_name}. "
            "Supported: 'gpt-4o-mini', 'ollama-llama3', 'ollama-llama3.2'"
        )


def _create_openai_generator(model: str, **kwargs) -> Any:
    """Create an OpenAI-compatible generator via CMU AI Gateway.
    
    Args:
        model: Model name (e.g., "gpt-4o-mini-2024-07-18")
        **kwargs: Additional arguments
        
    Returns:
        ChatOpenAI instance
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. Run: pip install langchain-openai"
        )
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "This is required for CMU AI Gateway access."
        )
    
    # Default to the specific versioned model if just "gpt-4o-mini" is passed
    if model == "gpt-4o-mini":
        model = "gpt-4o-mini-2024-07-18"
    
    print(f"[INFO] Initializing OpenAI Generator: {model}")
    
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://ai-gateway.andrew.cmu.edu/",
        temperature=kwargs.get("temperature", 0),
        max_tokens=kwargs.get("max_tokens", 1024),
    )
    
    return llm


def _create_ollama_generator(model: str, **kwargs) -> Any:
    """Create an Ollama-based local LLM generator.
    
    Args:
        model: Ollama model name (e.g., "llama3.2", "llama3", "mistral")
        **kwargs: Additional arguments
        
    Returns:
        ChatOllama instance
    """
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-community not installed. Run: pip install langchain-community"
        )
    
    # Get Ollama base URL (default to localhost)
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"[INFO] Initializing Ollama Generator: {model} at {ollama_base_url}")
    
    llm = ChatOllama(
        model=model,
        base_url=ollama_base_url,
        temperature=kwargs.get("temperature", 0),
        # Ollama-specific: timeout for generation
        timeout=kwargs.get("timeout", 300),
    )
    
    return llm
