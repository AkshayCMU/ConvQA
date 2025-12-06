# RUNBOOK: RAG System Execution Guide

**Course**: 11-697 Introduction to Question Answering with LLMs  
**Student**: Akshay Kulkarni (akshayk2)  
**GitHub**: https://github.com/AkshayCMU/11697-hw6-akshayk2

**Note**: This runbook and associated documentation were created with the assistance of AI tools (Claude/ChatGPT) and Grammarly for editing and organization.

---

## Quick Start

### Prerequisites
```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://ai-gateway.andrew.cmu.edu/"
```

### Optional: Install Ollama for local models
```bash
# Download and install Ollama from https://ollama.com
# Pull the Qwen model
ollama pull qwen2.5:1.5b
```

---

## Repository Structure (For Automated Grading)

```
11697-hw6-akshayk2/
├── README.md                          # Project documentation
├── RUNBOOK.md                         # This file
├── topic.txt                          # Topic description
├── requirements.txt                   # Python dependencies
├── data/
│   ├── question.tsv                   # 15 test questions (evaluated)
│   ├── answer.tsv                     # 15 ground truth answers
│   ├── evidence.tsv                   # 15 evidence sources
│   ├── corpus/                        # 410 document chunks
│   │   └── ITC_2025_*.txt
│   └── qa/                            # 100 generated questions (full set)
│       ├── question.tsv
│       ├── answer.tsv
│       └── evidence.tsv
├── src/
│   ├── rag.py                         # Main RAG pipeline
│   ├── evaluate.py                    # Evaluation script
│   ├── output_analysis.py             # Analysis & visualization
│   └── rag_system/                    # Modular components
│       ├── __init__.py
│       ├── pipeline.py
│       ├── retriever_factory.py
│       ├── generator_factory.py
│       └── data_loader.py
└── output/
    ├── prediction/                    # Model outputs (5 systems)
    │   ├── None_gpt-4o-mini.tsv
    │   ├── semantic-openai_gpt-4o-mini.tsv
    │   ├── semantic-openai_ollama-qwen2.5:1.5b.tsv
    │   ├── hybrid_gpt-4o-mini.tsv
    │   └── hybrid_ollama-qwen2.5:1.5b.tsv
    ├── evaluation/                    # Evaluation scores
    │   └── <system_name>.tsv
    └── analysis/                      # Charts and analysis
        └── *.png
```

---

## Step-by-Step Execution

### Step 1: Run RAG Systems (5 configurations)

```bash
# System 1: Baseline (No Retrieval) + GPT-4o-mini
python src/rag.py --retriever None --generator gpt-4o-mini

# System 2: Semantic Retrieval + GPT-4o-mini  
python src/rag.py --retriever semantic-openai --generator gpt-4o-mini

# System 3: Semantic Retrieval + Qwen (local)
python src/rag.py --retriever semantic-openai --generator ollama-qwen2.5:1.5b

# System 4: Hybrid Retrieval + GPT-4o-mini
python src/rag.py --retriever hybrid --generator gpt-4o-mini

# System 5: Hybrid Retrieval + Qwen (local)
python src/rag.py --retriever hybrid --generator ollama-qwen2.5:1.5b
```

**Expected Output**: Creates TSV files in `output/prediction/`

---

### Step 2: Run Evaluations

```bash
# Evaluate each system
python src/evaluate.py --prediction output/prediction/None_gpt-4o-mini.tsv
python src/evaluate.py --prediction output/prediction/semantic-openai_gpt-4o-mini.tsv
python src/evaluate.py --prediction output/prediction/semantic-openai_ollama-qwen2.5:1.5b.tsv
python src/evaluate.py --prediction output/prediction/hybrid_gpt-4o-mini.tsv
python src/evaluate.py --prediction output/prediction/hybrid_ollama-qwen2.5:1.5b.tsv
```

**Expected Output**: Creates TSV files in `output/evaluation/`

---

### Step 3: Generate Analysis & Visualizations

```bash
python src/output_analysis.py
```

**Expected Output**: Creates PNG charts and summary in `output/analysis/`

---

## Output Files

### Prediction Files (`output/prediction/`)
Format: `<retriever>_<generator>.tsv`

```
<prediction>\t<optional_metadata>
```

Example:
```
Quarterly and Half-Yearly	{"score": 0.95}
```

### Evaluation Files (`output/evaluation/`)
Format: `<system_name>.tsv`

```
<f1_score>\t<llm_judge_score>
```

Example:
```
0.85	4
```

### Analysis Files (`output/analysis/`)
- `llm_scores_by_system.png` - Overall comparison
- `f1_scores_by_system.png` - F1 score comparison  
- `heatmap_by_question_type.png` - Performance matrix
- `radar_chart.png` - Multi-dimensional view
- Individual question type breakdowns

---

## Key Components

### Retriever Options
- `None` - No retrieval (baseline)
- `semantic-openai` - Dense retrieval with OpenAI embeddings
- `hybrid` - BM25 + E5-large with RRF fusion

### Generator Options
- `gpt-4o-mini` - OpenAI model via CMU Gateway
- `ollama-qwen2.5:1.5b` - Local open-source model

### Evaluation Metrics
1. **F1 Score**: Token-level overlap
2. **LLM-as-a-Judge**: GPT-4o-mini rates 1-5

---

## One-Command Execution (All Systems)

```bash
# Run all 5 systems
for retriever in None semantic-openai hybrid; do
  for generator in gpt-4o-mini ollama-qwen2.5:1.5b; do
    # Skip invalid combinations
    if [ "$retriever" = "None" ] && [ "$generator" = "ollama-qwen2.5:1.5b" ]; then
      continue
    fi
    python src/rag.py --retriever $retriever --generator $generator
  done
done

# Run all evaluations
for pred in output/prediction/*.tsv; do
  python src/evaluate.py --prediction "$pred"
done

# Generate analysis
python src/output_analysis.py
```

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not found"
```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_API_BASE="https://ai-gateway.andrew.cmu.edu/"
```

### Issue: "Ollama model not found"
```bash
ollama pull qwen2.5:1.5b
ollama list  # Verify installation
```

### Issue: "ChromaDB error"
```bash
# Clear vector database cache
rm -rf data/chroma_db_*
# Re-run the system
```

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

---

## Notes for TAs

1. **QA Files Location**: 
   - `data/question.tsv`, `data/answer.tsv`, `data/evidence.tsv` = 15 test questions (used for evaluation)
   - `data/qa/` = 100 generated questions (complete set)

2. **Prediction Files**: All 5 required systems have outputs in `output/prediction/`

3. **Evaluation Files**: Corresponding evaluation results in `output/evaluation/`

4. **Corpus**: 410 text chunks under `data/corpus/ITC_2025_*.txt`

5. **System Naming Convention**: `<retriever>_<generator>.tsv`
   - Example: `hybrid_gpt-4o-mini.tsv`

---

## Contact

**Andrew ID**: akshayk2  
**GitHub**: https://github.com/AkshayCMU/11697-hw6-akshayk2

For any questions about replication, please refer to README.md or reach out via Canvas.
