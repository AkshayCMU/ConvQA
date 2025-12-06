# HW6 Final Report: RAG-based Question Answering System

**Course:** 95-702 Distributed Systems  
**Topic:** LLM-powered investor question answering over corporate annual reports  
**Domain:** ITC Limited Annual Report 2025

---

## 1. Introduction

This report presents the design, implementation, and evaluation of a Retrieval-Augmented Generation (RAG) system for answering investor questions over corporate annual reports. The system addresses the challenge of domain-specific question answering where large language models lack the specialized knowledge contained in financial documents.

### 1.1 Problem Statement

Investor queries about corporate performance require precise answers grounded in official documents. Pure LLMs suffer from:
- Lack of access to private/recent documents
- Hallucination of financial figures
- Inability to cite sources

RAG addresses these limitations by retrieving relevant document chunks before generation.

### 1.2 Scope

- **Corpus:** ITC Limited Annual Report 2025 (410 text chunks)
- **Questions:** 100 QA pairs covering factoid, MCQ, and list questions
- **Evaluation:** 15 questions across 5 system configurations

---

## 2. System Architecture

### 2.1 RAG Pipeline

```
Question → Retriever → Top-K Chunks → Prompt Construction → Generator → Answer
```

### 2.2 Components

| Component | Options Implemented |
|-----------|---------------------|
| **Retriever** | None (baseline), Semantic (OpenAI embeddings), Hybrid (BM25 + E5-large) |
| **Generator** | GPT-4o-mini (API), Qwen2.5:1.5b (local) |
| **Vector Store** | ChromaDB |
| **Fusion** | Reciprocal Rank Fusion (RRF) for hybrid retrieval |

### 2.3 Retriever Implementations

**Semantic Retriever:**
- Embedding model: OpenAI text-embedding-3-small
- Vector store: ChromaDB with cosine similarity
- Top-K: 5 chunks

**Hybrid Retriever:**
- Sparse: BM25 (keyword matching)
- Dense: E5-large-v2 embeddings
- Fusion: RRF with k=60
- Top-K: 5 chunks after fusion

### 2.4 Generator Implementations

**GPT-4o-mini:**
- Accessed via CMU LiteLLM Gateway
- Temperature: 0 for deterministic outputs
- Context window: ~128K tokens

**Qwen2.5:1.5b:**
- Local deployment via Ollama
- 1.5 billion parameters
- Chosen for resource efficiency

---

## 3. Evaluation Methodology

### 3.1 Metrics

**F1 Score (Token Overlap):**
```
Precision = |predicted ∩ ground_truth| / |predicted|
Recall = |predicted ∩ ground_truth| / |ground_truth|
F1 = 2 * Precision * Recall / (Precision + Recall)
```

**LLM-as-a-Judge:**
GPT-4o-mini rates answer quality on a 1-5 scale:
- 5: Perfect - accurate and complete
- 4: Good - mostly correct with minor issues
- 3: Partial - contains correct information but incomplete
- 2: Poor - mostly incorrect
- 1: Failed - completely wrong or no answer

### 3.2 Question Types

| Type | Count | Description |
|------|-------|-------------|
| Factoid | 6 (40%) | Single fact extraction |
| MCQ | 6 (40%) | Multiple choice selection |
| List | 3 (20%) | Multi-item enumeration |

---

## 4. Results

### 4.1 Overall System Performance

| Rank | System | LLM Judge | F1 Score |
|------|--------|-----------|----------|
| 1 | hybrid_gpt-4o-mini | **3.93** | **0.476** |
| 2 | semantic-openai_gpt-4o-mini | 3.27 | 0.304 |
| 3 | semantic-openai_ollama-qwen2.5:1.5b | 2.07 | 0.142 |
| 4 | hybrid_ollama-qwen2.5:1.5b | 1.93 | 0.131 |
| 5 | None_gpt-4o-mini (baseline) | 1.73 | 0.148 |

**Key Finding:** The best RAG system (hybrid + GPT-4o-mini) achieves a **127% improvement** over the baseline.

### 4.2 Performance by Question Type

| Question Type | Difficulty | Best System | Score | Observation |
|---------------|------------|-------------|-------|-------------|
| Factoid | Easy | hybrid_gpt-4o-mini | 4.00 | Direct lookup; all RAG systems perform well |
| MCQ | Moderate | hybrid_gpt-4o-mini | 4.33 | Requires reasoning over options |
| List | Hard | hybrid_gpt-4o-mini | 3.00 | Multi-hop aggregation; most systems fail |

**Analysis:**

1. **Factoid questions** are the easiest because they require locating a single piece of information. Even weaker models with good retrieval achieve reasonable scores (2.17-4.00). The baseline fails (1.50) because the LLM lacks domain-specific knowledge about ITC's annual report.

2. **MCQ questions** show the highest variance between systems. GPT-4o-mini excels at reasoning over multiple options to select the correct answer. Interestingly, the baseline occasionally succeeds (2.33) when the question touches on general knowledge that the LLM has from training.

3. **List questions** are the most challenging because they require aggregating multiple pieces of information scattered across the document. Only the hybrid + GPT-4o-mini combination achieves acceptable performance (3.00). All other systems essentially fail (score ≤ 1.33). This suggests that top-5 retrieval may not capture all required list items, and smaller models struggle with synthesis.

### 4.3 Ablation Study: Retriever Impact

| Retriever | LLM Avg | F1 Avg | Analysis |
|-----------|---------|--------|----------|
| Hybrid | 2.93 | 0.303 | Best overall performance |
| Semantic | 2.67 | 0.223 | Good for conceptual queries |
| None | 1.73 | 0.148 | Baseline; relies on parametric knowledge |

**Insight:** Hybrid retrieval outperforms pure semantic retrieval by 10% in LLM score. The combination of BM25 (exact keyword matching) and dense embeddings (semantic similarity) is particularly effective for financial documents that contain specific terminology, numbers, and proper nouns.

### 4.4 Ablation Study: Generator Impact

| Generator | LLM Avg | F1 Avg | Analysis |
|-----------|---------|--------|----------|
| GPT-4o-mini | 2.98 | 0.309 | Strong reasoning and instruction following |
| Qwen2.5:1.5b | 2.00 | 0.136 | Limited by model size |

**Insight:** Generator quality has a **49% impact** on performance. GPT-4o-mini's superior capabilities in reasoning, instruction following, and answer formatting are critical for achieving high scores. The 1.5B parameter Qwen model struggles with:
- Multi-hop reasoning
- Precise numerical extraction
- Consistent answer formatting

---

## 5. Error Analysis

### 5.1 Failure Cases

| Question | Type | Issue | Root Cause |
|----------|------|-------|------------|
| Q4: Undisputed dues values | List | Low scores across all systems | Numerical data in tables; requires precise extraction |
| Q6: Foreign ownership % | MCQ | Most systems failed | Specific percentage not in retrieved chunks |
| Q14: Stock options issued | Factoid | Low F1 despite high LLM | Format mismatch (different number representation) |

### 5.2 Success Patterns

| Pattern | Example | Why It Works |
|---------|---------|--------------|
| Direct factoid | Q12: Plastic neutrality year | Single fact, easily retrievable, clear answer |
| Clear MCQ | Q5: Subsidiary oversight | Retrieved text matches option verbatim |
| Keyword-rich | Q7: Key brands | BM25 excels at exact keyword matching |

### 5.3 Divergent Cases (High Variance)

Some questions showed high variance between systems:

- **Q1 (Variance: 2.64):** Investor complaints filing frequency
  - Hybrid retriever succeeded; semantic retriever failed
  - BM25 captured "quarterly" and "half-yearly" keywords

- **Q5 (Variance: 3.84):** Subsidiary oversight mechanism
  - RAG systems succeeded (score: 5); baseline failed completely (score: 1)
  - Demonstrates the value of retrieval for domain-specific questions

---

## 6. Discussion

### 6.1 Key Takeaways

1. **RAG is essential for domain-specific QA.** The baseline LLM without retrieval achieves only 1.73/5.0, while the best RAG system achieves 3.93/5.0—a 127% improvement.

2. **Retrieval quality matters more than generator size.** A good retriever with a weaker generator (hybrid + Qwen: 1.93) outperforms no retriever with a strong generator (none + GPT: 1.73).

3. **Hybrid retrieval is superior for financial documents.** The combination of BM25 and dense retrieval captures both exact matches (company names, figures) and semantic relationships (concepts, paraphrases).

4. **Question complexity determines performance ceiling.** Factoid questions are solved effectively by all RAG systems, but list questions remain challenging even for the best configuration.

### 6.2 Limitations

1. **Evaluation scale:** Only 15/100 questions evaluated due to compute constraints
2. **Local model size:** Qwen2.5:1.5b is small; larger models (7B, 70B) would likely perform better
3. **Chunk strategy:** Fixed-size chunking may split tables and relevant context
4. **No reranking:** Adding a cross-encoder reranker could improve retrieval precision

### 6.3 Future Work

1. **Larger local models:** Evaluate Qwen-7B, Llama-3-70B for better open-source performance
2. **Table-aware chunking:** Preserve table structure for financial data extraction
3. **Iterative retrieval:** Implement retrieval feedback for multi-hop questions
4. **Chain-of-thought prompting:** Improve reasoning for complex list questions

---

## 7. Conclusion

This project demonstrates that RAG significantly improves LLM performance on domain-specific question answering tasks. The best configuration—hybrid retrieval with GPT-4o-mini—achieves a 127% improvement over the baseline.

Key findings include:
- **Hybrid retrieval > Semantic > None** for financial document QA
- **GPT-4o-mini >> Qwen2.5:1.5b** due to superior reasoning capabilities
- **Question difficulty hierarchy:** Factoid (easy) → MCQ (moderate) → List (hard)
- **Retrieval is necessary but not sufficient:** Generator quality determines the ability to synthesize and format answers correctly

The system successfully answers investor questions about ITC's annual report, with particularly strong performance on factoid and MCQ questions. List questions remain challenging and represent an opportunity for future improvement through better chunking strategies and multi-hop retrieval mechanisms.

---

## 8. References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval.
3. Wang, L., et al. (2024). "E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training." ACL.
4. LangChain Documentation: https://python.langchain.com/
5. ChromaDB Documentation: https://docs.trychroma.com/

---

## Appendix A: System Configurations

| System ID | Retriever | Embedding Model | Generator | Top-K |
|-----------|-----------|-----------------|-----------|-------|
| None_gpt-4o-mini | None | - | GPT-4o-mini | - |
| semantic-openai_gpt-4o-mini | Semantic | text-embedding-3-small | GPT-4o-mini | 5 |
| semantic-openai_ollama-qwen2.5:1.5b | Semantic | text-embedding-3-small | Qwen2.5:1.5b | 5 |
| hybrid_gpt-4o-mini | Hybrid | E5-large-v2 + BM25 | GPT-4o-mini | 5 |
| hybrid_ollama-qwen2.5:1.5b | Hybrid | E5-large-v2 + BM25 | Qwen2.5:1.5b | 5 |

## Appendix B: Generated Visualizations

The following visualizations are available in `output/analysis/`:

1. `llm_scores_by_system.png` - Overall system comparison
2. `f1_scores_by_system.png` - F1 score comparison
3. `heatmap_by_question_type.png` - Performance matrix
4. `radar_chart.png` - Multi-dimensional comparison
5. `llm_scores_factoid.png` - Factoid question breakdown
6. `llm_scores_mcq.png` - MCQ question breakdown
7. `llm_scores_list.png` - List question breakdown
