"""
Evaluation script for RAG systems.

Metrics:
1. F1 Score (Overlap)
2. LLM-as-a-Judge (GPT-4o-mini rating 1-5)

Usage:
    python src/evaluate.py --prediction_file output/prediction/None_gpt-4o-mini.tsv --output_dir output/evaluation
"""

import argparse
import os
import string
import collections
import re
import time
from pathlib import Path
from typing import List, Dict

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG system outputs")
    parser.add_argument("--prediction_file", type=str, required=True, help="Path to prediction TSV")
    parser.add_argument("--gold_file", type=str, default="data/qa/answer.tsv", help="Path to gold answers TSV")
    parser.add_argument("--output_dir", type=str, default="output/evaluation", help="Directory to save evaluation results")
    return parser.parse_args()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def llm_judge(gold, pred, llm):
    """Rate answer 1-5 using LLM."""
    prompt = f"""
    Rate the quality of the system prediction compared to the gold answer on a scale of 1-5.
    1: Completely incorrect
    2: Mostly incorrect
    3: Partially correct (covers some key points)
    4: Mostly correct (minor details missing)
    5: Fully correct and complete

    Gold Answer: {gold}
    System Prediction: {pred}

    Return ONLY the number (1-5).
    """
    try:
        res = llm.invoke(prompt)
        score = res.content.strip()
        # Extract first digit found
        match = re.search(r'\d', score)
        if match:
            return int(match.group())
        return 1
    except:
        return 1

def evaluate_file(pred_file, gold_file, output_dir):
    print(f"[INFO] Evaluating {pred_file}...")
    
    # Setup LLM
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key: 
        raise RuntimeError("OPENAI_API_KEY required for LLM judge")
        
    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/',
        temperature=0
    )

    # Read lines
    with open(pred_file, 'r', encoding='utf-8') as f:
        preds = [line.strip() for line in f]
        
    with open(gold_file, 'r', encoding='utf-8') as f:
        golds = [line.strip() for line in f]
        
    if len(preds) != len(golds):
        print(f"[WARN] Mismatch in line counts: Preds={len(preds)}, Golds={len(golds)}")
        # Truncate to min length
        min_len = min(len(preds), len(golds))
        preds = preds[:min_len]
        golds = golds[:min_len]

    f1_scores = []
    judge_scores = []
    
    predictions_out = []
    
    total = len(preds)
    for i, (p, g) in enumerate(zip(preds, golds)):
        if (i+1) % 10 == 0:
            print(f"  Eval {i+1}/{total}...")
            
        f1 = compute_f1(g, p)
        judge = llm_judge(g, p, llm)
        
        f1_scores.append(f1)
        judge_scores.append(judge)
        
        predictions_out.append(f"{judge}\t{f1}")

    # Save results
    path = Path(pred_file)
    out_name = path.name
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / out_name, 'w', encoding='utf-8') as f:
        for line in predictions_out:
            f.write(f"{line}\n")
            
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_judge = sum(judge_scores) / len(judge_scores)
    
    print(f"[RESULT] {out_name}: Avg F1={avg_f1:.3f}, Avg Judge={avg_judge:.2f}")

def main():
    args = parse_args()
    evaluate_file(args.prediction_file, args.gold_file, args.output_dir)

if __name__ == "__main__":
    main()
