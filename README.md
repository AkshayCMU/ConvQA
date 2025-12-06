# HW6: RAG-based Question Answering System

**Course**: 95-702 Distributed Systems  
**Topic**: LLM-powered investor question answering over corporate annual reports  
**Domain**: ITC Limited Annual Report 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Systems Evaluated](#systems-evaluated)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results and Analysis](#results-and-analysis)
8. [Key Findings](#key-findings)
9. [Error Analysis](#error-analysis)
10. [Runbook](#runbook)

---

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering investor questions over ITC Limited's 2025 Annual Report. The system compares multiple retrieval strategies (none, semantic, hybrid) and language models (GPT-4o-mini, Qwen2.5:1.5b) to evaluate their effectiveness across different question types.

### Corpus Statistics
- **Document**: ITC Limited Annual Report 2025
- **Chunks**: 410 text segments
- **QA Pairs**: 100 questions (15 evaluated due to compute constraints)
- **Question Types**: Factoid (40%), MCQ (40%), List (20%)

---

## Project Structure

```
hw6_submission/
├── data/
│   ├── corpus/                 # 410 document chunks
│   │   └── ITC_2025_p*.txt
│   └── qa/                     # Question-answer pairs
│       ├── question.tsv        # Questions with type annotations
│       ├── answer.tsv          # Ground truth answers
│       └── evidence.tsv        # Evidence passages
├── src/
│   ├── rag.py                  # Main RAG pipeline entry point
│   ├── evaluate.py             # Evaluation metrics (F1, LLM Judge)
│   ├── output_analysis.py      # Analysis and visualization
│   └── rag_system/             # Modular RAG components
│       ├── __init__.py
│       ├── pipeline.py         # RAG/Baseline pipeline logic
│       ├── retriever_factory.py # Retriever implementations
│       ├── generator_factory.py # LLM generator implementations
│       └── data_loader.py      # Data loading utilities
├── output/
│   ├── prediction/             # Model predictions (TSV)
│   ├── evaluation/             # Evaluation scores (TSV)
│   └── analysis/               # Charts and reports
├── topic.txt                   # Topic description
├── requirements.txt            # Python dependencies
├── run_analysis.sh             # Analysis runner script
└── README.md                   # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (via CMU Gateway)
- Ollama (for local Qwen model, optional)

### Setup

```bash
# Clone and navigate to project
cd hw6_submission

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://cmu.litellm.ai"
```

---

## Usage

### Running the RAG Pipeline

```bash
# Baseline (no retrieval)
python src/rag.py --retriever None --generator gpt-4o-mini

# Semantic retriever with GPT-4o-mini
python src/rag.py --retriever semantic-openai --generator gpt-4o-mini

# Hybrid retriever with GPT-4o-mini
python src/rag.py --retriever hybrid --generator gpt-4o-mini

# Hybrid retriever with local Qwen
python src/rag.py --retriever hybrid --generator ollama-qwen2.5:1.5b
```

### Running Evaluation

```bash
python src/evaluate.py --prediction output/prediction/hybrid_gpt-4o-mini.tsv
```

### Running Analysis

```bash
./run_analysis.sh
```

---

## Systems Evaluated

| System ID | Retriever | Generator | Description |
|-----------|-----------|-----------|-------------|
| None_gpt-4o-mini | None | GPT-4o-mini | Baseline - LLM only, no retrieval |
| semantic-openai_gpt-4o-mini | Semantic | GPT-4o-mini | Dense retrieval with OpenAI embeddings |
| semantic-openai_ollama-qwen2.5:1.5b | Semantic | Qwen2.5:1.5b | Dense retrieval with local LLM |
| hybrid_gpt-4o-mini | Hybrid | GPT-4o-mini | BM25 + E5-large with RRF fusion |
| hybrid_ollama-qwen2.5:1.5b | Hybrid | Qwen2.5:1.5b | Hybrid retrieval with local LLM |

### Retriever Details

- **None**: No context retrieval; LLM relies on parametric knowledge
- **Semantic (OpenAI)**: Dense retrieval using `text-embedding-3-small`, ChromaDB vector store, top-5 chunks
- **Hybrid**: Combines BM25 sparse retrieval with E5-large-v2 dense embeddings using Reciprocal Rank Fusion (RRF)

### Generator Details

- **GPT-4o-mini**: OpenAI model accessed via CMU LiteLLM Gateway
- **Qwen2.5:1.5b**: Open-source model running locally via Ollama

---

## Evaluation Metrics

### 1. F1 Score (Token Overlap)
Measures lexical overlap between predicted and ground truth answers at the token level.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### 2. LLM-as-a-Judge
GPT-4o-mini rates answer quality on a 1-5 scale:

| Score | Description |
|-------|-------------|
| 5 | Perfect - Accurate and complete |
| 4 | Good - Mostly correct with minor omissions |
| 3 | Partial - Contains correct information but incomplete |
| 2 | Poor - Mostly incorrect or irrelevant |
| 1 | Failed - Completely wrong or "I don't know" |

---

## Results and Analysis

### Overall System Performance

| Rank | System | LLM Judge (Avg) | LLM Judge (Std) | F1 (Avg) | F1 (Std) |
|------|--------|-----------------|-----------------|----------|----------|
| 1 | hybrid_gpt-4o-mini | **3.93** | 1.61 | **0.476** | 0.364 |
| 2 | semantic-openai_gpt-4o-mini | 3.27 | 1.88 | 0.304 | 0.337 |
| 3 | semantic-openai_ollama-qwen2.5:1.5b | 2.07 | 1.57 | 0.142 | 0.247 |
| 4 | hybrid_ollama-qwen2.5:1.5b | 1.93 | 1.57 | 0.131 | 0.198 |
| 5 | None_gpt-4o-mini (baseline) | 1.73 | 1.48 | 0.148 | 0.339 |

### Performance by Question Type

| Question Type | Count | Best System | LLM Score | Analysis |
|---------------|-------|-------------|-----------|----------|
| **Factoid** | 6 (40%) | hybrid_gpt-4o-mini | 4.00 | Relatively easier; requires locating specific facts |
| **MCQ** | 6 (40%) | hybrid_gpt-4o-mini | 4.33 | Requires understanding context to select correct option |
| **List** | 3 (20%) | hybrid_gpt-4o-mini | 3.00 | Most challenging; requires aggregating multiple pieces of information |

#### Detailed Question Type Analysis

**Factoid Questions (Avg LLM Score by System):**
| System | Score | Observation |
|--------|-------|-------------|
| hybrid_gpt-4o-mini | 4.00 | Strong retrieval helps locate specific facts |
| semantic-openai_gpt-4o-mini | 3.83 | Dense retrieval effective for factual queries |
| hybrid_ollama-qwen2.5:1.5b | 2.33 | Small model struggles with precise extraction |
| semantic-openai_ollama-qwen2.5:1.5b | 2.17 | Context helps but generation quality limits performance |
| None_gpt-4o-mini | 1.50 | Without retrieval, LLM lacks domain knowledge |

**MCQ Questions (Avg LLM Score by System):**
| System | Score | Observation |
|--------|-------|-------------|
| hybrid_gpt-4o-mini | 4.33 | Best performance; hybrid retrieval finds relevant context |
| semantic-openai_gpt-4o-mini | 3.67 | Good but occasionally retrieves less relevant chunks |
| semantic-openai_ollama-qwen2.5:1.5b | 2.50 | Local model can identify correct option with context |
| None_gpt-4o-mini | 2.33 | Some MCQs answerable from parametric knowledge |
| hybrid_ollama-qwen2.5:1.5b | 1.83 | Retrieval helps but small model inconsistent |

**List Questions (Avg LLM Score by System):**
| System | Score | Observation |
|--------|-------|-------------|
| hybrid_gpt-4o-mini | 3.00 | Best but still challenging |
| semantic-openai_gpt-4o-mini | 1.33 | Struggles to aggregate multiple items |
| hybrid_ollama-qwen2.5:1.5b | 1.33 | Complex reasoning beyond model capacity |
| None_gpt-4o-mini | 1.00 | Complete failure without context |
| semantic-openai_ollama-qwen2.5:1.5b | 1.00 | Cannot synthesize information |

### Retriever Comparison

| Retriever | LLM Avg | F1 Avg | Observations |
|-----------|---------|--------|--------------|
| **Hybrid** | 2.93 | 0.303 | Best overall; BM25 handles keyword matching, dense handles semantic similarity |
| **Semantic** | 2.67 | 0.223 | Good for conceptual queries; may miss exact keyword matches |
| **None** | 1.73 | 0.148 | Baseline; LLM relies on parametric knowledge only |

### Generator Comparison

| Generator | LLM Avg | F1 Avg | Observations |
|-----------|---------|--------|--------------|
| **GPT-4o-mini** | 2.98 | 0.309 | Strong reasoning and generation capabilities |
| **Qwen2.5:1.5b** | 2.00 | 0.136 | Limited by model size; struggles with complex reasoning |

---

## Key Findings

### 1. RAG Provides Significant Improvement
- **127% improvement** in LLM Judge score compared to baseline
- Hybrid retrieval with GPT-4o-mini achieves 3.93/5.0 vs baseline 1.73/5.0
- Retrieval is essential for domain-specific questions

### 2. Question Difficulty Varies by Type
- **Factoid**: Easiest - all RAG systems perform reasonably well (2.17-4.00)
- **MCQ**: Moderate - requires understanding context to eliminate wrong options
- **List**: Hardest - requires aggregating information from multiple sources

### 3. Retriever Choice Matters More Than Generator
- Hybrid retriever improves scores even with weaker Qwen model
- Better retrieval compensates partially for weaker generation
- BM25 + dense hybrid captures both lexical and semantic matches

### 4. Generator Quality Has Significant Impact
- GPT-4o-mini outperforms Qwen2.5:1.5b by ~49% (2.98 vs 2.00)
- Larger models better at reasoning over retrieved context
- Small models struggle with multi-hop reasoning and synthesis

### 5. Variance Analysis
- High variance between systems on same questions indicates sensitivity to retrieval quality
- Some questions show perfect scores (5.0) while others fail completely (1.0)
- Divergent cases often involve ambiguous or complex multi-part questions

---

## Error Analysis

### Failure Cases

| Question | Type | Issue | Root Cause |
|----------|------|-------|------------|
| Q4: List undisputed dues values | List | All systems scored low | Requires precise numerical extraction from tables |
| Q6: Foreign ownership percentage | MCQ | Most systems failed | Specific percentage not in retrieved chunks |
| Q14: Stock options issued | Factoid | Low F1 despite high LLM | Answer format mismatch |

### Success Patterns

| Pattern | Example | Why It Works |
|---------|---------|--------------|
| Direct factoid | Q12: Plastic neutrality year | Single fact, easily retrievable |
| Clear MCQ | Q5: Subsidiary oversight | Options match retrieved text closely |
| Keyword-rich | Q7: Key brands launched | BM25 excels at exact keyword matching |

### Divergent Cases (High Variance)

| Question | Variance | Analysis |
|----------|----------|----------|
| Q1: Investor complaints filing frequency | 2.64 | Semantic retriever failed; hybrid succeeded |
| Q5: Subsidiary oversight mechanism | 3.84 | Baseline failed completely; RAG systems succeeded |

---

## Runbook

### Complete Execution Steps

```bash
# Step 1: Environment Setup
cd hw6_submission
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 2: Set API Keys
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="https://cmu.litellm.ai"

# Step 3: Run All RAG Configurations
python src/rag.py --retriever None --generator gpt-4o-mini
python src/rag.py --retriever semantic-openai --generator gpt-4o-mini
python src/rag.py --retriever semantic-openai --generator ollama-qwen2.5:1.5b
python src/rag.py --retriever hybrid --generator gpt-4o-mini
python src/rag.py --retriever hybrid --generator ollama-qwen2.5:1.5b

# Step 4: Run Evaluations
for pred in output/prediction/*.tsv; do
    python src/evaluate.py --prediction "$pred"
done

# Step 5: Generate Analysis
./run_analysis.sh

# Step 6: View Results
open output/analysis/*.png  # macOS
cat output/analysis/analysis_summary.txt
```

### Output Files

| Directory | Files | Description |
|-----------|-------|-------------|
| `output/prediction/` | `*.tsv` | Raw model predictions |
| `output/evaluation/` | `*.tsv` | LLM Judge + F1 scores per question |
| `output/analysis/` | `*.png`, `*.txt` | Visualizations and summary |

### Visualization Files

| File | Description |
|------|-------------|
| `llm_scores_by_system.png` | Overall system comparison |
| `f1_scores_by_system.png` | F1 score comparison |
| `heatmap_by_question_type.png` | Performance matrix by question type |
| `radar_chart.png` | Multi-dimensional system comparison |
| `llm_scores_factoid.png` | Factoid question breakdown |
| `llm_scores_mcq.png` | MCQ question breakdown |
| `llm_scores_list.png` | List question breakdown |

---

## Architecture Details

### RAG Pipeline Flow

```
Question → Retriever → Top-K Chunks → Prompt Construction → Generator → Answer
                ↓
        [None | Semantic | Hybrid]
                                                    ↓
                                            [GPT-4o-mini | Qwen2.5]
```

### Hybrid Retriever (RRF Fusion)

```python
# Reciprocal Rank Fusion
RRF_score(d) = Σ 1/(k + rank_i(d))

# Where k=60, combining:
# - BM25 sparse retrieval ranks
# - E5-large dense retrieval ranks
```

---

## Limitations and Future Work

1. **Evaluation Scale**: Only 15/100 questions evaluated due to compute constraints
2. **Local Model Size**: Qwen2.5:1.5b is small; larger models would likely perform better
3. **Chunk Strategy**: Fixed chunking may split relevant context
4. **No Reranking**: Adding a reranker could improve retrieval precision

---

## References

- LangChain Documentation: https://python.langchain.com/
- ChromaDB: https://docs.trychroma.com/
- OpenAI API: https://platform.openai.com/docs/
- Ollama: https://ollama.ai/
