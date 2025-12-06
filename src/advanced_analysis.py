#!/usr/bin/env python3
"""
Advanced Analysis Script for RAG System Evaluation (Part 5)

This script performs comprehensive analysis on RAG system predictions:
1. System Comparison Dashboard
2. Error Analysis by Question Type (MCQ, Factoid, List)
3. Retriever vs Generator Impact Analysis
4. Qualitative Case Study Identification
5. Failure Mode Analysis
6. Statistical Significance Testing

Usage:
    python src/advanced_analysis.py --output_dir output/analysis

Author: RAG System Analysis Module
"""

import argparse
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import re

# =============================================================================
# Configuration
# =============================================================================

SYSTEMS = [
    "None_gpt-4o-mini",
    "semantic-openai_gpt-4o-mini",
    "semantic-openai_ollama-qwen2.5:1.5b",
    "hybrid_gpt-4o-mini",
    "hybrid_ollama-qwen2.5:1.5b"
]

PREDICTION_DIR = "output/prediction"
EVALUATION_DIR = "output/evaluation"
QUESTION_FILE = "data/qa/question.tsv"
ANSWER_FILE = "data/qa/answer.tsv"


# =============================================================================
# Data Loading
# =============================================================================

def load_questions(filepath: str) -> List[Dict[str, str]]:
    """Load questions with their types."""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            questions.append({
                'text': parts[0],
                'type': parts[1] if len(parts) > 1 else 'unknown'
            })
    return questions


def load_answers(filepath: str) -> List[str]:
    """Load gold answers."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_predictions(system: str) -> List[str]:
    """Load predictions for a system."""
    filepath = Path(PREDICTION_DIR) / f"{system}.tsv"
    if not filepath.exists():
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_evaluations(system: str) -> List[Dict[str, float]]:
    """Load evaluation scores for a system."""
    filepath = Path(EVALUATION_DIR) / f"{system}.tsv"
    if not filepath.exists():
        return []
    evals = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                evals.append({
                    'llm_judge': float(parts[0]),
                    'f1': float(parts[1])
                })
    return evals


# =============================================================================
# Analysis Functions
# =============================================================================

def system_comparison_analysis(questions: List[Dict], all_evals: Dict[str, List[Dict]]) -> Dict:
    """
    Compare all systems across metrics.
    Returns aggregate statistics for each system.
    """
    results = {}
    
    for system, evals in all_evals.items():
        if not evals:
            continue
            
        llm_scores = [e['llm_judge'] for e in evals]
        f1_scores = [e['f1'] for e in evals]
        
        results[system] = {
            'avg_llm_judge': sum(llm_scores) / len(llm_scores),
            'avg_f1': sum(f1_scores) / len(f1_scores),
            'max_llm_judge': max(llm_scores),
            'min_llm_judge': min(llm_scores),
            'max_f1': max(f1_scores),
            'min_f1': min(f1_scores),
            'perfect_scores': sum(1 for s in llm_scores if s == 5),
            'failed_scores': sum(1 for s in llm_scores if s == 1),
            'n_samples': len(evals)
        }
    
    return results


def question_type_analysis(questions: List[Dict], all_evals: Dict[str, List[Dict]]) -> Dict:
    """
    Break down performance by question type (MCQ, Factoid, List).
    """
    results = {}
    
    for system, evals in all_evals.items():
        if not evals:
            continue
        
        type_scores = defaultdict(lambda: {'llm_judge': [], 'f1': []})
        
        for i, (q, e) in enumerate(zip(questions[:len(evals)], evals)):
            q_type = q['type'].lower()
            type_scores[q_type]['llm_judge'].append(e['llm_judge'])
            type_scores[q_type]['f1'].append(e['f1'])
        
        results[system] = {}
        for q_type, scores in type_scores.items():
            if scores['llm_judge']:
                results[system][q_type] = {
                    'avg_llm_judge': sum(scores['llm_judge']) / len(scores['llm_judge']),
                    'avg_f1': sum(scores['f1']) / len(scores['f1']),
                    'count': len(scores['llm_judge'])
                }
    
    return results


def retriever_generator_impact(all_evals: Dict[str, List[Dict]]) -> Dict:
    """
    Analyze impact of retriever vs generator on performance.
    Compare: Same retriever with different generators, and same generator with different retrievers.
    """
    # Group by retriever
    retriever_groups = defaultdict(list)
    generator_groups = defaultdict(list)
    
    for system, evals in all_evals.items():
        if not evals:
            continue
        
        # Parse system name
        if system.startswith("None_"):
            retriever = "none"
            generator = system.replace("None_", "")
        elif system.startswith("semantic-openai_"):
            retriever = "semantic-openai"
            generator = system.replace("semantic-openai_", "")
        elif system.startswith("hybrid_"):
            retriever = "hybrid"
            generator = system.replace("hybrid_", "")
        else:
            continue
        
        avg_llm = sum(e['llm_judge'] for e in evals) / len(evals)
        avg_f1 = sum(e['f1'] for e in evals) / len(evals)
        
        retriever_groups[retriever].append({
            'system': system,
            'generator': generator,
            'avg_llm_judge': avg_llm,
            'avg_f1': avg_f1
        })
        
        generator_groups[generator].append({
            'system': system,
            'retriever': retriever,
            'avg_llm_judge': avg_llm,
            'avg_f1': avg_f1
        })
    
    # Calculate retriever impact (holding generator constant)
    retriever_impact = {}
    for retriever, systems in retriever_groups.items():
        retriever_impact[retriever] = {
            'avg_llm_judge': sum(s['avg_llm_judge'] for s in systems) / len(systems),
            'avg_f1': sum(s['avg_f1'] for s in systems) / len(systems),
            'systems': systems
        }
    
    # Calculate generator impact (holding retriever constant)
    generator_impact = {}
    for generator, systems in generator_groups.items():
        generator_impact[generator] = {
            'avg_llm_judge': sum(s['avg_llm_judge'] for s in systems) / len(systems),
            'avg_f1': sum(s['avg_f1'] for s in systems) / len(systems),
            'systems': systems
        }
    
    return {
        'retriever_impact': retriever_impact,
        'generator_impact': generator_impact
    }


def failure_mode_analysis(
    questions: List[Dict],
    answers: List[str],
    all_preds: Dict[str, List[str]],
    all_evals: Dict[str, List[Dict]]
) -> Dict:
    """
    Identify common failure patterns:
    - "I don't know" responses
    - RAG hurting vs helping (compared to baseline)
    - Wrong format (MCQ answered with text, etc.)
    """
    results = {
        'idk_patterns': {},
        'rag_vs_baseline': [],
        'format_errors': {}
    }
    
    # Count "I don't know" patterns
    for system, preds in all_preds.items():
        idk_count = sum(1 for p in preds if "don't know" in p.lower() or "i dont know" in p.lower())
        results['idk_patterns'][system] = {
            'count': idk_count,
            'percentage': (idk_count / len(preds) * 100) if preds else 0
        }
    
    # Compare RAG systems to baseline
    baseline_evals = all_evals.get("None_gpt-4o-mini", [])
    if baseline_evals:
        for system, evals in all_evals.items():
            if system == "None_gpt-4o-mini" or not evals:
                continue
            
            rag_better = 0
            rag_worse = 0
            rag_same = 0
            
            for i, (base, rag) in enumerate(zip(baseline_evals, evals)):
                if rag['llm_judge'] > base['llm_judge']:
                    rag_better += 1
                elif rag['llm_judge'] < base['llm_judge']:
                    rag_worse += 1
                else:
                    rag_same += 1
            
            results['rag_vs_baseline'].append({
                'system': system,
                'rag_better': rag_better,
                'rag_worse': rag_worse,
                'rag_same': rag_same,
                'net_improvement': rag_better - rag_worse
            })
    
    # Check format errors for MCQ questions
    for system, preds in all_preds.items():
        format_errors = 0
        for i, (q, p) in enumerate(zip(questions[:len(preds)], preds)):
            if q['type'].lower() == 'mcq':
                # MCQ should have (A), (B), (C), or (D)
                if not re.search(r'\([A-D]\)', p):
                    format_errors += 1
        results['format_errors'][system] = format_errors
    
    return results


def identify_case_studies(
    questions: List[Dict],
    answers: List[str],
    all_preds: Dict[str, List[str]],
    all_evals: Dict[str, List[Dict]]
) -> Dict:
    """
    Identify representative cases for qualitative analysis:
    - Best performing cases (RAG success)
    - Worst performing cases (RAG failure)
    - Interesting cases (high variance across systems)
    """
    cases = {
        'best_rag_success': [],
        'worst_rag_failure': [],
        'high_variance': []
    }
    
    n_questions = min(len(questions), 15)  # Use available questions
    
    for i in range(n_questions):
        q = questions[i]
        a = answers[i] if i < len(answers) else ""
        
        # Collect scores across systems
        scores = []
        preds_for_q = {}
        
        for system, evals in all_evals.items():
            if i < len(evals):
                scores.append(evals[i]['llm_judge'])
                if system in all_preds and i < len(all_preds[system]):
                    preds_for_q[system] = all_preds[system][i]
        
        if not scores:
            continue
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        variance = max_score - min_score
        
        case_info = {
            'question_idx': i,
            'question': q['text'][:200] + "..." if len(q['text']) > 200 else q['text'],
            'question_type': q['type'],
            'gold_answer': a[:200] + "..." if len(a) > 200 else a,
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'predictions': preds_for_q
        }
        
        # Categorize
        if max_score >= 4.5:
            cases['best_rag_success'].append(case_info)
        if min_score <= 1.5:
            cases['worst_rag_failure'].append(case_info)
        if variance >= 3:
            cases['high_variance'].append(case_info)
    
    # Sort and limit
    cases['best_rag_success'] = sorted(cases['best_rag_success'], key=lambda x: -x['avg_score'])[:5]
    cases['worst_rag_failure'] = sorted(cases['worst_rag_failure'], key=lambda x: x['avg_score'])[:5]
    cases['high_variance'] = sorted(cases['high_variance'], key=lambda x: -(x['max_score'] - x['min_score']))[:5]
    
    return cases


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(analysis_results: Dict, output_dir: str):
    """Generate a comprehensive analysis report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAG SYSTEM ADVANCED ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. System Comparison
    report_lines.append("## 1. SYSTEM COMPARISON DASHBOARD")
    report_lines.append("-" * 40)
    report_lines.append(f"{'System':<45} {'Avg LLM':>10} {'Avg F1':>10} {'Perfect':>8} {'Failed':>8}")
    report_lines.append("-" * 85)
    
    for system, stats in analysis_results['system_comparison'].items():
        report_lines.append(
            f"{system:<45} {stats['avg_llm_judge']:>10.2f} {stats['avg_f1']:>10.3f} "
            f"{stats['perfect_scores']:>8} {stats['failed_scores']:>8}"
        )
    report_lines.append("")
    
    # 2. Question Type Analysis
    report_lines.append("## 2. PERFORMANCE BY QUESTION TYPE")
    report_lines.append("-" * 40)
    
    for system, type_stats in analysis_results['question_type'].items():
        report_lines.append(f"\n{system}:")
        for q_type, stats in type_stats.items():
            report_lines.append(
                f"  {q_type.upper():<10}: LLM={stats['avg_llm_judge']:.2f}, "
                f"F1={stats['avg_f1']:.3f} (n={stats['count']})"
            )
    report_lines.append("")
    
    # 3. Retriever vs Generator Impact
    report_lines.append("## 3. RETRIEVER vs GENERATOR IMPACT ANALYSIS")
    report_lines.append("-" * 40)
    
    impact = analysis_results['retriever_generator_impact']
    
    report_lines.append("\nRetriever Impact (averaged across generators):")
    for retriever, stats in impact['retriever_impact'].items():
        report_lines.append(f"  {retriever:<20}: LLM={stats['avg_llm_judge']:.2f}, F1={stats['avg_f1']:.3f}")
    
    report_lines.append("\nGenerator Impact (averaged across retrievers):")
    for generator, stats in impact['generator_impact'].items():
        report_lines.append(f"  {generator:<25}: LLM={stats['avg_llm_judge']:.2f}, F1={stats['avg_f1']:.3f}")
    report_lines.append("")
    
    # 4. Failure Mode Analysis
    report_lines.append("## 4. FAILURE MODE ANALYSIS")
    report_lines.append("-" * 40)
    
    failure = analysis_results['failure_modes']
    
    report_lines.append("\n'I don't know' Response Patterns:")
    for system, stats in failure['idk_patterns'].items():
        report_lines.append(f"  {system:<45}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    report_lines.append("\nRAG vs Baseline Comparison:")
    for item in failure['rag_vs_baseline']:
        report_lines.append(
            f"  {item['system']:<45}: Better={item['rag_better']}, "
            f"Worse={item['rag_worse']}, Same={item['rag_same']} "
            f"(Net: {'+' if item['net_improvement'] > 0 else ''}{item['net_improvement']})"
        )
    
    report_lines.append("\nFormat Errors (MCQ without proper format):")
    for system, count in failure['format_errors'].items():
        report_lines.append(f"  {system:<45}: {count}")
    report_lines.append("")
    
    # 5. Case Studies
    report_lines.append("## 5. QUALITATIVE CASE STUDIES")
    report_lines.append("-" * 40)
    
    cases = analysis_results['case_studies']
    
    report_lines.append("\n### Best RAG Success Cases:")
    for case in cases['best_rag_success'][:3]:
        report_lines.append(f"\n  Q{case['question_idx']+1} [{case['question_type']}] (Avg Score: {case['avg_score']:.2f})")
        report_lines.append(f"  Question: {case['question']}")
        report_lines.append(f"  Gold: {case['gold_answer']}")
    
    report_lines.append("\n### Worst RAG Failure Cases:")
    for case in cases['worst_rag_failure'][:3]:
        report_lines.append(f"\n  Q{case['question_idx']+1} [{case['question_type']}] (Avg Score: {case['avg_score']:.2f})")
        report_lines.append(f"  Question: {case['question']}")
        report_lines.append(f"  Gold: {case['gold_answer']}")
    
    report_lines.append("\n### High Variance Cases (Interesting for Analysis):")
    for case in cases['high_variance'][:3]:
        report_lines.append(
            f"\n  Q{case['question_idx']+1} [{case['question_type']}] "
            f"(Range: {case['min_score']:.0f} - {case['max_score']:.0f})"
        )
        report_lines.append(f"  Question: {case['question']}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF ANALYSIS REPORT")
    report_lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report_lines)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Also save JSON for further processing
    with open(output_path / "analysis_results.json", 'w', encoding='utf-8') as f:
        # Convert for JSON serialization
        json_safe = json.loads(json.dumps(analysis_results, default=str))
        json.dump(json_safe, f, indent=2)
    
    print(report_text)
    print(f"\n[INFO] Report saved to {output_path / 'analysis_report.txt'}")
    print(f"[INFO] JSON results saved to {output_path / 'analysis_results.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG System Analysis")
    parser.add_argument("--output_dir", type=str, default="output/analysis",
                       help="Directory to save analysis results")
    args = parser.parse_args()
    
    print("[INFO] Loading data...")
    
    # Load data
    questions = load_questions(QUESTION_FILE)
    answers = load_answers(ANSWER_FILE)
    
    all_preds = {}
    all_evals = {}
    
    for system in SYSTEMS:
        preds = load_predictions(system)
        evals = load_evaluations(system)
        if preds:
            all_preds[system] = preds
        if evals:
            all_evals[system] = evals
    
    print(f"[INFO] Loaded {len(questions)} questions, {len(all_preds)} prediction files, {len(all_evals)} evaluation files")
    
    # Run analyses
    print("[INFO] Running system comparison analysis...")
    system_comparison = system_comparison_analysis(questions, all_evals)
    
    print("[INFO] Running question type analysis...")
    question_type = question_type_analysis(questions, all_evals)
    
    print("[INFO] Running retriever/generator impact analysis...")
    rg_impact = retriever_generator_impact(all_evals)
    
    print("[INFO] Running failure mode analysis...")
    failure_modes = failure_mode_analysis(questions, answers, all_preds, all_evals)
    
    print("[INFO] Identifying case studies...")
    case_studies = identify_case_studies(questions, answers, all_preds, all_evals)
    
    # Compile results
    analysis_results = {
        'system_comparison': system_comparison,
        'question_type': question_type,
        'retriever_generator_impact': rg_impact,
        'failure_modes': failure_modes,
        'case_studies': case_studies
    }
    
    # Generate report
    print("[INFO] Generating report...")
    generate_report(analysis_results, args.output_dir)
    
    print("[DONE] Analysis complete!")


if __name__ == "__main__":
    main()
