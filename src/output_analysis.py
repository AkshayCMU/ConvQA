#!/usr/bin/env python3
"""
RAG System Output Analysis
Generates performance metrics and visualizations by question type.
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_questions(path: str) -> List[Tuple[str, str]]:
    """Load questions with their types from TSV file."""
    questions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                questions.append((parts[0], parts[1]))
    return questions


def load_answers(path: str) -> List[str]:
    """Load ground truth answers from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_predictions(path: str) -> List[str]:
    """Load model predictions from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_evaluations(path: str) -> List[Tuple[int, float]]:
    """Load evaluation scores (LLM Judge, F1) from TSV file."""
    scores = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    llm_score = int(parts[0])
                    f1_score = float(parts[1])
                    scores.append((llm_score, f1_score))
                except ValueError:
                    continue
    return scores


def print_header(text: str, char: str = "="):
    """Print formatted section header."""
    width = 80
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.END}\n")


def print_subheader(text: str):
    """Print formatted subsection header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─' * 60}{Colors.END}")


def print_table(headers: List[str], rows: List[List], highlight_best: bool = True):
    """Print formatted ASCII table with optional best-value highlighting."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    col_widths = [w + 2 for w in col_widths]
    
    header_line = "|".join(f"{h:^{col_widths[i]}}" for i, h in enumerate(headers))
    separator = "+".join("-" * w for w in col_widths)
    top_border = "+".join("-" * w for w in col_widths)
    
    print(f"+{top_border}+")
    print(f"|{Colors.BOLD}{header_line}{Colors.END}|")
    print(f"+{separator}+")
    
    best_values = {}
    if highlight_best and rows:
        for col_idx in range(1, len(headers)):
            try:
                values = [float(str(row[col_idx]).rstrip('%')) for row in rows if row[col_idx] not in ['N/A', '-']]
                if values:
                    best_values[col_idx] = max(values)
            except ValueError:
                pass
    
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cell_str = str(cell)
            try:
                cell_val = float(cell_str.rstrip('%'))
                if i in best_values and abs(cell_val - best_values[i]) < 0.001:
                    cell_str = f"{Colors.GREEN}{Colors.BOLD}{cell_str}{Colors.END}"
            except ValueError:
                pass
            cells.append(f"{cell_str:^{col_widths[i]}}" if Colors.GREEN not in cell_str else f"  {cell_str}  ")
        print(f"|{'|'.join(cells)}|")
    
    print(f"+{top_border}+")


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate descriptive statistics for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5
    
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4)
    }


def create_bar_chart(data: Dict[str, float], title: str, filename: str, output_dir: str):
    """Create ASCII bar chart and save matplotlib version if available."""
    print(f"\n{Colors.BOLD}  {title}{Colors.END}")
    print(f"  {'─' * 50}")
    
    max_val = max(data.values()) if data.values() else 1
    bar_width = 40
    
    for label, value in sorted(data.items(), key=lambda x: -x[1]):
        bar_len = int((value / max_val) * bar_width) if max_val > 0 else 0
        bar = "#" * bar_len + "." * (bar_width - bar_len)
        color = Colors.GREEN if value == max(data.values()) else Colors.BLUE
        print(f"  {label:30} {color}{bar}{Colors.END} {value:.3f}")
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = list(data.keys())
        values = list(data.values())
        
        colors = ['#2ecc71' if v == max(values) else '#3498db' for v in values]
        bars = ax.barh(labels, values, color=colors)
        
        ax.set_xlabel('Score')
        ax.set_title(title)
        ax.set_xlim(0, max(values) * 1.1 if values else 1)
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {Colors.GREEN}[OK] Saved: {filepath}{Colors.END}")
    except ImportError:
        print(f"  {Colors.YELLOW}[WARN] matplotlib not installed - skipping image export{Colors.END}")


def analyze_by_question_type(questions: List[Tuple[str, str]], 
                             evaluations: Dict[str, List[Tuple[int, float]]]) -> Dict:
    """Aggregate results by question type."""
    results = defaultdict(lambda: defaultdict(lambda: {"llm": [], "f1": []}))
    
    for system, scores in evaluations.items():
        for i, (llm_score, f1_score) in enumerate(scores):
            if i < len(questions):
                q_type = questions[i][1]
                results[q_type][system]["llm"].append(llm_score)
                results[q_type][system]["f1"].append(f1_score)
    
    return results


def analyze_success_failure(questions: List[Tuple[str, str]], 
                           evaluations: Dict[str, List[Tuple[int, float]]],
                           predictions: Dict[str, List[str]],
                           answers: List[str]) -> Dict:
    """Identify success, failure, and high-variance cases."""
    cases = {"success": [], "failure": [], "divergent": []}
    
    for i in range(len(questions)):
        if i >= len(answers):
            continue
            
        scores_for_q = {}
        for system, evals in evaluations.items():
            if i < len(evals):
                scores_for_q[system] = evals[i]
        
        if not scores_for_q:
            continue
        
        llm_scores = [s[0] for s in scores_for_q.values()]
        f1_scores = [s[1] for s in scores_for_q.values()]
        
        avg_llm = sum(llm_scores) / len(llm_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        variance = sum((s - avg_llm) ** 2 for s in llm_scores) / len(llm_scores)
        
        case_info = {
            "idx": i + 1,
            "question": questions[i][0][:80] + "..." if len(questions[i][0]) > 80 else questions[i][0],
            "type": questions[i][1],
            "avg_llm": round(avg_llm, 2),
            "avg_f1": round(avg_f1, 3),
            "variance": round(variance, 2),
            "ground_truth": answers[i][:60] + "..." if len(answers[i]) > 60 else answers[i],
            "predictions": {sys: (predictions[sys][i][:60] + "..." if len(predictions[sys][i]) > 60 else predictions[sys][i]) 
                          for sys in predictions if i < len(predictions[sys])}
        }
        
        if avg_llm >= 4.0 and avg_f1 >= 0.5:
            cases["success"].append(case_info)
        elif avg_llm <= 2.0 or avg_f1 <= 0.2:
            cases["failure"].append(case_info)
        elif variance >= 2.0:
            cases["divergent"].append(case_info)
    
    return cases


def print_case_analysis(cases: Dict):
    """Print detailed case analysis for success, failure, and divergent cases."""
    print_subheader("SUCCESS CASES (Avg LLM >= 4.0, F1 >= 0.5)")
    if cases["success"]:
        for case in cases["success"][:3]:
            print(f"\n  {Colors.GREEN}Q{case['idx']}{Colors.END} [{case['type']}] LLM: {case['avg_llm']}, F1: {case['avg_f1']}")
            print(f"  {Colors.CYAN}Question:{Colors.END} {case['question']}")
            print(f"  {Colors.CYAN}Ground Truth:{Colors.END} {case['ground_truth']}")
    else:
        print(f"  {Colors.YELLOW}No clear success cases found.{Colors.END}")
    
    print_subheader("FAILURE CASES (Avg LLM <= 2.0 or F1 <= 0.2)")
    if cases["failure"]:
        for case in cases["failure"][:3]:
            print(f"\n  {Colors.RED}Q{case['idx']}{Colors.END} [{case['type']}] LLM: {case['avg_llm']}, F1: {case['avg_f1']}")
            print(f"  {Colors.CYAN}Question:{Colors.END} {case['question']}")
            print(f"  {Colors.CYAN}Ground Truth:{Colors.END} {case['ground_truth']}")
    else:
        print(f"  {Colors.GREEN}No complete failures found.{Colors.END}")
    
    print_subheader("DIVERGENT CASES (High Variance Between Systems)")
    if cases["divergent"]:
        for case in cases["divergent"][:3]:
            print(f"\n  {Colors.YELLOW}Q{case['idx']}{Colors.END} [{case['type']}] Variance: {case['variance']}")
            print(f"  {Colors.CYAN}Question:{Colors.END} {case['question']}")
            print(f"  {Colors.CYAN}Predictions:{Colors.END}")
            for sys, pred in case['predictions'].items():
                print(f"    - {sys}: {pred}")
    else:
        print(f"  {Colors.BLUE}All systems performed consistently.{Colors.END}")


def create_heatmap(results: Dict, output_dir: str):
    """Create heatmap comparing systems across question types."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        
        q_types = sorted(results.keys())
        systems = sorted(list(results[q_types[0]].keys()) if q_types else [])
        
        if not systems or not q_types:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax_idx, (metric, title) in enumerate([("llm", "LLM Judge Scores"), ("f1", "F1 Scores")]):
            data = []
            for q_type in q_types:
                row = []
                for sys in systems:
                    vals = results[q_type][sys][metric]
                    row.append(sum(vals) / len(vals) if vals else 0)
                data.append(row)
            
            data = np.array(data)
            im = axes[ax_idx].imshow(data, cmap='RdYlGn', aspect='auto')
            
            axes[ax_idx].set_xticks(range(len(systems)))
            axes[ax_idx].set_xticklabels([s.replace('_', '\n') for s in systems], fontsize=8, rotation=45, ha='right')
            axes[ax_idx].set_yticks(range(len(q_types)))
            axes[ax_idx].set_yticklabels(q_types)
            axes[ax_idx].set_title(title)
            
            for i in range(len(q_types)):
                for j in range(len(systems)):
                    axes[ax_idx].text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)
            
            plt.colorbar(im, ax=axes[ax_idx])
        
        plt.suptitle('Performance by Question Type', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'heatmap_by_question_type.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {Colors.GREEN}[OK] Saved: {filepath}{Colors.END}")
        
    except ImportError:
        print(f"  {Colors.YELLOW}[WARN] matplotlib/numpy not installed - skipping heatmap{Colors.END}")


def create_radar_chart(results: Dict, output_dir: str):
    """Create radar chart comparing system performance across question types."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        
        q_types = sorted(results.keys())
        if not q_types:
            return
            
        systems = sorted(list(results[q_types[0]].keys()))
        
        angles = np.linspace(0, 2 * np.pi, len(q_types), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, sys in enumerate(systems):
            values = []
            for q_type in q_types:
                vals = results[q_type][sys]["llm"]
                values.append(sum(vals) / len(vals) if vals else 0)
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=sys.replace('_', ' '), color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(q_types)
        ax.set_ylim(0, 5)
        ax.set_title('System Performance by Question Type\n(LLM Judge Score)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'radar_chart.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {Colors.GREEN}[OK] Saved: {filepath}{Colors.END}")
        
    except ImportError:
        print(f"  {Colors.YELLOW}[WARN] matplotlib/numpy not installed - skipping radar chart{Colors.END}")


def generate_summary_report(evaluations: Dict, questions: List, output_dir: str):
    """Generate text summary report."""
    report_path = os.path.join(output_dir, 'analysis_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HW6 RAG SYSTEM - ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Questions Evaluated: {len(questions)}\n")
        f.write(f"Systems Compared: {len(evaluations)}\n\n")
        
        type_counts = defaultdict(int)
        for _, q_type in questions:
            type_counts[q_type] += 1
        
        f.write("Question Type Distribution:\n")
        for q_type, count in sorted(type_counts.items()):
            f.write(f"  - {q_type}: {count} ({count/len(questions)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("SYSTEM RANKINGS\n")
        f.write("-" * 40 + "\n")
        
        system_scores = {}
        for sys, evals in evaluations.items():
            llm_scores = [e[0] for e in evals]
            f1_scores = [e[1] for e in evals]
            system_scores[sys] = {
                "llm_avg": sum(llm_scores) / len(llm_scores) if llm_scores else 0,
                "f1_avg": sum(f1_scores) / len(f1_scores) if f1_scores else 0
            }
        
        f.write("\nBy LLM Judge Score:\n")
        for rank, (sys, scores) in enumerate(sorted(system_scores.items(), key=lambda x: -x[1]["llm_avg"]), 1):
            f.write(f"  {rank}. {sys}: {scores['llm_avg']:.3f}\n")
        
        f.write("\nBy F1 Score:\n")
        for rank, (sys, scores) in enumerate(sorted(system_scores.items(), key=lambda x: -x[1]["f1_avg"]), 1):
            f.write(f"  {rank}. {sys}: {scores['f1_avg']:.3f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  {Colors.GREEN}[OK] Saved: {report_path}{Colors.END}")


def main():
    """Main analysis pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'qa')
    eval_dir = os.path.join(base_dir, 'output', 'evaluation')
    pred_dir = os.path.join(base_dir, 'output', 'prediction')
    output_dir = os.path.join(base_dir, 'output', 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print_header("HW6 RAG SYSTEM - OUTPUT ANALYSIS")
    
    print(f"{Colors.BOLD}Loading data...{Colors.END}")
    questions = load_questions(os.path.join(data_dir, 'question.tsv'))
    answers = load_answers(os.path.join(data_dir, 'answer.tsv'))
    
    evaluations = {}
    predictions = {}
    
    eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.tsv')]
    for eval_file in eval_files:
        system_name = eval_file.replace('.tsv', '')
        evaluations[system_name] = load_evaluations(os.path.join(eval_dir, eval_file))
        
        pred_file = os.path.join(pred_dir, eval_file)
        if os.path.exists(pred_file):
            predictions[system_name] = load_predictions(pred_file)
    
    print(f"  Loaded {len(questions)} questions")
    print(f"  Loaded {len(evaluations)} system evaluations")
    print(f"  Question types: {set(q[1] for q in questions)}")
    
    # Section 1: Overall System Comparison
    print_header("SECTION 1: OVERALL SYSTEM COMPARISON")
    
    headers = ["System", "LLM Judge (Avg)", "LLM Judge (Std)", "F1 (Avg)", "F1 (Std)"]
    rows = []
    
    for sys_name, evals in sorted(evaluations.items()):
        llm_scores = [e[0] for e in evals]
        f1_scores = [e[1] for e in evals]
        
        llm_stats = calculate_statistics(llm_scores)
        f1_stats = calculate_statistics(f1_scores)
        
        rows.append([
            sys_name,
            f"{llm_stats['mean']:.3f}",
            f"{llm_stats['std']:.3f}",
            f"{f1_stats['mean']:.3f}",
            f"{f1_stats['std']:.3f}"
        ])
    
    print_table(headers, rows)
    
    llm_avgs = {sys: sum(e[0] for e in evals)/len(evals) for sys, evals in evaluations.items()}
    f1_avgs = {sys: sum(e[1] for e in evals)/len(evals) for sys, evals in evaluations.items()}
    
    create_bar_chart(llm_avgs, "Average LLM Judge Score by System", "llm_scores_by_system.png", output_dir)
    create_bar_chart(f1_avgs, "Average F1 Score by System", "f1_scores_by_system.png", output_dir)
    
    # Section 2: Analysis by Question Type
    print_header("SECTION 2: PERFORMANCE BY QUESTION TYPE")
    
    results_by_type = analyze_by_question_type(questions, evaluations)
    
    for q_type in sorted(results_by_type.keys()):
        print_subheader(f"Question Type: {q_type.upper()}")
        
        type_count = sum(1 for q in questions if q[1] == q_type)
        print(f"  Count: {type_count} questions\n")
        
        headers = ["System", "LLM Avg", "LLM Std", "F1 Avg", "F1 Std"]
        rows = []
        
        for sys_name in sorted(results_by_type[q_type].keys()):
            data = results_by_type[q_type][sys_name]
            llm_stats = calculate_statistics(data["llm"])
            f1_stats = calculate_statistics(data["f1"])
            
            rows.append([
                sys_name,
                f"{llm_stats['mean']:.3f}",
                f"{llm_stats['std']:.3f}",
                f"{f1_stats['mean']:.3f}",
                f"{f1_stats['std']:.3f}"
            ])
        
        print_table(headers, rows)
        
        type_llm_avgs = {sys: calculate_statistics(data["llm"])["mean"] 
                        for sys, data in results_by_type[q_type].items()}
        create_bar_chart(type_llm_avgs, f"LLM Judge Score - {q_type.upper()} Questions", 
                        f"llm_scores_{q_type}.png", output_dir)
    
    print_subheader("Generating Comparison Visualizations")
    create_heatmap(results_by_type, output_dir)
    create_radar_chart(results_by_type, output_dir)
    
    # Section 3: Retriever vs Generator Ablation
    print_header("SECTION 3: RETRIEVER VS GENERATOR ABLATION")
    
    retriever_performance = defaultdict(lambda: {"llm": [], "f1": []})
    generator_performance = defaultdict(lambda: {"llm": [], "f1": []})
    
    for sys_name, evals in evaluations.items():
        parts = sys_name.split('_')
        if len(parts) >= 2:
            retriever = parts[0]
            generator = '_'.join(parts[1:])
        else:
            retriever = sys_name
            generator = "unknown"
        
        for llm, f1 in evals:
            retriever_performance[retriever]["llm"].append(llm)
            retriever_performance[retriever]["f1"].append(f1)
            generator_performance[generator]["llm"].append(llm)
            generator_performance[generator]["f1"].append(f1)
    
    print_subheader("Performance by Retriever Type")
    headers = ["Retriever", "LLM Avg", "LLM Std", "F1 Avg", "F1 Std", "Count"]
    rows = []
    for ret_name in sorted(retriever_performance.keys()):
        data = retriever_performance[ret_name]
        llm_stats = calculate_statistics(data["llm"])
        f1_stats = calculate_statistics(data["f1"])
        rows.append([ret_name, f"{llm_stats['mean']:.3f}", f"{llm_stats['std']:.3f}",
                    f"{f1_stats['mean']:.3f}", f"{f1_stats['std']:.3f}", len(data["llm"])])
    print_table(headers, rows)
    
    print_subheader("Performance by Generator Type")
    headers = ["Generator", "LLM Avg", "LLM Std", "F1 Avg", "F1 Std", "Count"]
    rows = []
    for gen_name in sorted(generator_performance.keys()):
        data = generator_performance[gen_name]
        llm_stats = calculate_statistics(data["llm"])
        f1_stats = calculate_statistics(data["f1"])
        rows.append([gen_name, f"{llm_stats['mean']:.3f}", f"{llm_stats['std']:.3f}",
                    f"{f1_stats['mean']:.3f}", f"{f1_stats['std']:.3f}", len(data["llm"])])
    print_table(headers, rows)
    
    # Section 4: Case Analysis
    print_header("SECTION 4: CASE ANALYSIS")
    
    cases = analyze_success_failure(questions, evaluations, predictions, answers)
    print_case_analysis(cases)
    
    # Section 5: Question-by-Question Breakdown
    print_header("SECTION 5: QUESTION-BY-QUESTION BREAKDOWN")
    
    headers = ["Q#", "Type", "Best System", "Best LLM", "Best F1"]
    rows = []
    
    for i, (question, q_type) in enumerate(questions):
        best_sys = None
        best_llm = 0
        best_f1 = 0
        
        for sys_name, evals in evaluations.items():
            if i < len(evals):
                llm, f1 = evals[i]
                if llm > best_llm or (llm == best_llm and f1 > best_f1):
                    best_llm = llm
                    best_f1 = f1
                    best_sys = sys_name
        
        rows.append([f"Q{i+1}", q_type, best_sys if best_sys else "N/A", str(best_llm), f"{best_f1:.3f}"])
    
    print_table(headers, rows, highlight_best=False)
    
    # Section 6: Key Findings
    print_header("SECTION 6: KEY FINDINGS")
    
    best_system = max(evaluations.keys(), key=lambda x: sum(e[0] for e in evaluations[x])/len(evaluations[x]))
    best_llm_avg = sum(e[0] for e in evaluations[best_system]) / len(evaluations[best_system])
    
    baseline_key = [k for k in evaluations.keys() if k.startswith('None')]
    if baseline_key:
        baseline_avg = sum(e[0] for e in evaluations[baseline_key[0]]) / len(evaluations[baseline_key[0]])
        improvement = ((best_llm_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
    else:
        improvement = 0
    
    print(f"""
{Colors.BOLD}KEY FINDINGS:{Colors.END}

  {Colors.GREEN}1. Best Performing System:{Colors.END}
     {best_system}
     Average LLM Judge Score: {best_llm_avg:.3f}/5.0

  {Colors.GREEN}2. RAG Improvement over Baseline:{Colors.END}
     {improvement:.1f}% improvement in LLM Judge Score

  {Colors.GREEN}3. Question Type Insights:{Colors.END}""")
    
    for q_type in sorted(results_by_type.keys()):
        best_for_type = max(results_by_type[q_type].keys(),
                          key=lambda x: calculate_statistics(results_by_type[q_type][x]["llm"])["mean"])
        best_score = calculate_statistics(results_by_type[q_type][best_for_type]["llm"])["mean"]
        print(f"     {q_type.upper()}: Best = {best_for_type} ({best_score:.2f})")
    
    print(f"""
  {Colors.GREEN}4. Retriever Impact:{Colors.END}""")
    best_retriever = max(retriever_performance.keys(),
                        key=lambda x: calculate_statistics(retriever_performance[x]["llm"])["mean"])
    print(f"     Best Retriever: {best_retriever}")
    
    print(f"""
  {Colors.GREEN}5. Generator Impact:{Colors.END}""")
    best_generator = max(generator_performance.keys(),
                        key=lambda x: calculate_statistics(generator_performance[x]["llm"])["mean"])
    print(f"     Best Generator: {best_generator}")
    
    print_subheader("Generating Summary Report")
    generate_summary_report(evaluations, questions, output_dir)
    
    print_header("ANALYSIS COMPLETE")
    print(f"""
{Colors.BOLD}Output directory:{Colors.END} {output_dir}/

{Colors.CYAN}Generated Files:{Colors.END}
  - llm_scores_by_system.png
  - f1_scores_by_system.png
  - llm_scores_factoid.png
  - llm_scores_mcq.png
  - llm_scores_list.png
  - heatmap_by_question_type.png
  - radar_chart.png
  - analysis_summary.txt
""")


if __name__ == "__main__":
    main()
