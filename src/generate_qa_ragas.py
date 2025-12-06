#!/usr/bin/env python3
"""
Advanced Synthetic QA Generation using RAGAS TestsetGenerator.

This script generates high-quality evaluation datasets with:
- Factoid questions (simple factual recall)
- List questions (enumerate multiple items)
- MCQ questions (multiple choice WITH options included)

Requirements:
    pip install ragas langchain langchain-openai langchain-community

Usage:
    python src/generate_qa_ragas.py --corpus_dir data/corpus --output_dir data/qa --num_questions 100
    
For Colab:
    !python src/generate_qa_ragas.py --corpus_dir data/corpus --output_dir data/qa --num_questions 100
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re

# LangChain imports
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RAGAS imports
try:
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    RAGAS_AVAILABLE = True
except Exception as e:
    RAGAS_AVAILABLE = False
    print(f"Warning: RAGAS not available ({e}). Will use fallback LLM-based generation.")


def load_corpus(corpus_dir: str) -> List[Document]:
    """Load all text chunks from corpus directory into LangChain Documents."""
    corpus_path = Path(corpus_dir)
    documents = []
    
    txt_files = sorted(corpus_path.glob("*.txt"))
    print(f"Found {len(txt_files)} text files in {corpus_dir}")
    
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if content:
            # Extract PDF name from chunk filename (e.g., ITC_2025_p1-p3_chunk001.txt -> ITC_2025)
            filename = txt_file.name
            parts = filename.split('_')
            if len(parts) >= 2:
                pdf_name = f"{parts[0]}_{parts[1]}"  # e.g., ITC_2025
            else:
                pdf_name = filename.replace('.txt', '')
            
            doc = Document(
                page_content=content,
                metadata={
                    'filename': filename,
                    'source': str(txt_file),
                    'pdf_name': pdf_name
                }
            )
            documents.append(doc)
    
    print(f"Loaded {len(documents)} documents with content")
    return documents


def generate_with_ragas(
    documents: List[Document],
    num_questions: int,
    api_key: str,
    api_base: str
) -> List[Dict]:
    """Generate QA pairs using RAGAS TestsetGenerator."""
    
    # Initialize OpenAI models for RAGAS
    generator_llm = ChatOpenAI(
        model="gpt-5",
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.7
    )
    
    critic_llm = ChatOpenAI(
        model="gpt-5",
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.2
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    
    # Create TestsetGenerator
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings
    )
    
    # Generate with distribution favoring variety
    # We'll generate more than needed to ensure we can filter/transform enough
    target_count = int(num_questions * 1.5)  # Generate 50% more for filtering
    
    print(f"Generating {target_count} raw questions with RAGAS...")
    
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        test_size=target_count,
        distributions={
            simple: 0.4,      # Will become factoid
            reasoning: 0.35,  # Will become list or mcq
            multi_context: 0.25  # Will become list or mcq
        },
        raise_exceptions=False
    )
    
    # Convert to list of dicts
    qa_pairs = []
    for row in testset.to_pandas().itertuples():
        qa_pairs.append({
            'question': row.question,
            'answer': row.ground_truth,
            'contexts': row.contexts if hasattr(row, 'contexts') else [],
            'evolution_type': row.evolution_type if hasattr(row, 'evolution_type') else 'simple',
            'metadata': row.metadata if hasattr(row, 'metadata') else {}
        })
    
    print(f"Generated {len(qa_pairs)} raw QA pairs")
    return qa_pairs


def generate_with_llm_direct(
    documents: List[Document],
    num_questions: int,
    api_key: str,
    api_base: str
) -> List[Dict]:
    """Generate QA pairs directly with LLM (High Quality)."""
    
    # User requested 'gpt-5' or better than mini
    model_name = "gpt-5" 
    print(f"Using model: {model_name} for generation")
    
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.7
    )
    
    # Sample documents to use - ensure we cover the corpus
    sample_size = min(50, len(documents))
    # Sort by filename to ensure deterministic sampling or just shuffle
    random.shuffle(documents)
    sampled_docs = documents[:sample_size]
    
    qa_pairs = []
    # Generate slightly more than needed per doc to buffer
    questions_per_doc = max(3, int(num_questions / sample_size * 1.5))
    
    prompt = ChatPromptTemplate.from_template("""
You are an expert at creating evaluation datasets for QA systems.
Based on the following text, generate {n} high-quality question-answer pairs.

Text Chunk:
{context}

Requirements:
1. Generate diverse question types:
   - "simple": Factual questions (What is X?)
   - "reasoning": Questions requiring inference or synthesis
   - "list": Questions asking for a list of items (List the factors...)
2. Questions must be answerable purely from the text.
3. Answers must be precise and accurate.

Output ONLY a valid JSON array of objects:
[
    {{"question": "Question text...", "answer": "Answer text...", "type": "simple"}},
    {{"question": "Question text...", "answer": "Answer text...", "type": "list"}}
]
""")
    
    chain = prompt | llm | StrOutputParser()
    
    print(f"Generating questions from {sample_size} documents...")
    
    for i, doc in enumerate(sampled_docs):
        if len(qa_pairs) >= num_questions * 1.5:
            break
            
        try:
            result = chain.invoke({
                "context": doc.page_content[:3000],  # Limit context size
                "n": questions_per_doc
            })
            
            # Parse JSON
            result = result.strip()
            if result.startswith("```"):
                result = re.sub(r'^```json?\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            
            pairs = json.loads(result)
            
            for pair in pairs:
                qa_pairs.append({
                    'question': pair['question'],
                    'answer': pair['answer'],
                    'contexts': [doc.page_content],
                    'evolution_type': pair.get('type', 'simple'),
                    'metadata': {'filename': doc.metadata.get('filename', '')}
                })
            
            print(f"  Processed {i+1}/{sample_size} documents, {len(qa_pairs)} QA pairs so far")
            
        except Exception as e:
            print(f"  Error processing document {i}: {e}")
            continue
    
    return qa_pairs


def transform_to_mcq(
    question: str,
    answer: str,
    context: str,
    llm: ChatOpenAI
) -> Tuple[str, str]:
    """Transform a question into MCQ format with options included."""
    
    prompt = ChatPromptTemplate.from_template("""
Convert the following question into a multiple choice question (MCQ).

Original Question: {question}
Correct Answer: {answer}
Context: {context}

Requirements:
1. Create 4 options (A, B, C, D)
2. One option must be the correct answer
3. Other options should be plausible but incorrect (distractors)
4. Options should be based on the context when possible

Output format (exactly this structure):
QUESTION: <rewritten question text>? (A) <option1> (B) <option2> (C) <option3> (D) <option4>
ANSWER: <letter of correct answer>
CORRECT_TEXT: <text of correct answer>

Only output in this exact format, nothing else.
""")
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context[:2000]
        })
        
        # Parse result
        lines = result.strip().split('\n')
        mcq_question = ""
        mcq_answer = ""
        
        for line in lines:
            if line.startswith("QUESTION:"):
                mcq_question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                letter = line.replace("ANSWER:", "").strip()
            elif line.startswith("CORRECT_TEXT:"):
                mcq_answer = line.replace("CORRECT_TEXT:", "").strip()
        
        if mcq_question and mcq_answer:
            return mcq_question, mcq_answer
        
    except Exception as e:
        print(f"MCQ conversion error: {e}")
    
    return None, None


def transform_to_list(
    question: str,
    answer: str,
    context: str,
    llm: ChatOpenAI
) -> Tuple[str, str]:
    """Transform a question into list format."""
    
    prompt = ChatPromptTemplate.from_template("""
Convert the following question into a list question that asks to enumerate multiple items.

Original Question: {question}
Original Answer: {answer}
Context: {context}

Requirements:
1. Rewrite the question to ask for a list of items (e.g., "List...", "What are the...", "Name the...")
2. The answer should be a list of items
3. If the original answer is a single item, expand to include related items from context

Output format (exactly this structure):
QUESTION: <rewritten list question>
ANSWER: <item1>, <item2>, <item3>, ...

Only output in this exact format, nothing else.
""")
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context[:2000]
        })
        
        lines = result.strip().split('\n')
        list_question = ""
        list_answer = ""
        
        for line in lines:
            if line.startswith("QUESTION:"):
                list_question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                list_answer = line.replace("ANSWER:", "").strip()
        
        if list_question and list_answer:
            return list_question, list_answer
            
    except Exception as e:
        print(f"List conversion error: {e}")
    
    return None, None


def process_qa_pairs(
    raw_pairs: List[Dict],
    num_questions: int,
    api_key: str,
    api_base: str
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Process raw QA pairs into three types: factoid, list, mcq.
    Ensures at least 20 of each type.
    """
    
    min_per_type = 20
    target_total = num_questions
    
    # Initialize LLM for transformations
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=0.5
    )
    
    factoid_pairs = []
    list_pairs = []
    mcq_pairs = []
    
    # Shuffle for variety
    random.shuffle(raw_pairs)
    
    print(f"\nProcessing {len(raw_pairs)} raw pairs into typed questions...")
    print(f"Target: {min_per_type}+ factoid, {min_per_type}+ list, {min_per_type}+ mcq")
    
    # First pass: categorize by evolution type
    simple_pairs = []
    reasoning_pairs = []
    multi_context_pairs = []
    
    for pair in raw_pairs:
        evo_type = pair.get('evolution_type', 'simple')
        if 'simple' in str(evo_type).lower():
            simple_pairs.append(pair)
        elif 'multi' in str(evo_type).lower():
            multi_context_pairs.append(pair)
        else:
            reasoning_pairs.append(pair)
    
    print(f"Raw distribution: {len(simple_pairs)} simple, {len(reasoning_pairs)} reasoning, {len(multi_context_pairs)} multi_context")
    
    # Convert simple -> factoid (straightforward mapping)
    print("\n1. Converting simple questions to factoid...")
    for pair in simple_pairs[:min(40, len(simple_pairs))]:  # Take up to 40 factoid
        factoid_pairs.append({
            'question': pair['question'],
            'answer': pair['answer'],
            'type': 'factoid',
            'contexts': pair.get('contexts', []),
            'metadata': pair.get('metadata', {})
        })
    
    print(f"   Created {len(factoid_pairs)} factoid questions")
    
    # Convert reasoning/multi_context -> list questions
    print("\n2. Converting reasoning questions to list format...")
    list_candidates = reasoning_pairs + multi_context_pairs
    converted_list = 0
    
    for pair in list_candidates:
        if converted_list >= 35:  # Target ~35 list questions
            break
            
        context = pair.get('contexts', [''])[0] if pair.get('contexts') else ''
        
        new_q, new_a = transform_to_list(
            pair['question'],
            pair['answer'],
            context,
            llm
        )
        
        if new_q and new_a:
            list_pairs.append({
                'question': new_q,
                'answer': new_a,
                'type': 'list',
                'contexts': pair.get('contexts', []),
                'metadata': pair.get('metadata', {})
            })
            converted_list += 1
            print(f"   List {converted_list}: {new_q[:60]}...")
    
    print(f"   Created {len(list_pairs)} list questions")
    
    # Convert some remaining to MCQ
    print("\n3. Converting questions to MCQ format...")
    mcq_candidates = [p for p in simple_pairs[40:]] + [p for p in reasoning_pairs if p not in list_candidates[:35]]
    converted_mcq = 0
    
    for pair in mcq_candidates:
        if converted_mcq >= 35:  # Target ~35 MCQ
            break
            
        context = pair.get('contexts', [''])[0] if pair.get('contexts') else ''
        
        new_q, new_a = transform_to_mcq(
            pair['question'],
            pair['answer'],
            context,
            llm
        )
        
        if new_q and new_a:
            # Verify MCQ has options
            if '(A)' in new_q and '(B)' in new_q:
                mcq_pairs.append({
                    'question': new_q,
                    'answer': new_a,
                    'type': 'mcq',
                    'contexts': pair.get('contexts', []),
                    'metadata': pair.get('metadata', {})
                })
                converted_mcq += 1
                print(f"   MCQ {converted_mcq}: {new_q[:60]}...")
    
    print(f"   Created {len(mcq_pairs)} MCQ questions")
    
    # Check if we need more of any type
    remaining_pairs = [p for p in raw_pairs if p not in simple_pairs[:40] + list_candidates[:35] + mcq_candidates[:35]]
    
    # Fill in any shortfalls
    if len(factoid_pairs) < min_per_type:
        needed = min_per_type - len(factoid_pairs)
        print(f"\n4. Need {needed} more factoid questions...")
        for pair in remaining_pairs[:needed]:
            factoid_pairs.append({
                'question': pair['question'],
                'answer': pair['answer'],
                'type': 'factoid',
                'contexts': pair.get('contexts', []),
                'metadata': pair.get('metadata', {})
            })
    
    if len(list_pairs) < min_per_type:
        needed = min_per_type - len(list_pairs)
        print(f"\n4. Need {needed} more list questions...")
        for pair in remaining_pairs[:(min_per_type - len(list_pairs))]:
            context = pair.get('contexts', [''])[0] if pair.get('contexts') else ''
            new_q, new_a = transform_to_list(pair['question'], pair['answer'], context, llm)
            if new_q:
                list_pairs.append({
                    'question': new_q,
                    'answer': new_a,
                    'type': 'list',
                    'contexts': pair.get('contexts', []),
                    'metadata': pair.get('metadata', {})
                })
    
    if len(mcq_pairs) < min_per_type:
        needed = min_per_type - len(mcq_pairs)
        print(f"\n4. Need {needed} more MCQ questions...")
        for pair in remaining_pairs[:(min_per_type - len(mcq_pairs))]:
            context = pair.get('contexts', [''])[0] if pair.get('contexts') else ''
            new_q, new_a = transform_to_mcq(pair['question'], pair['answer'], context, llm)
            if new_q and '(A)' in new_q:
                mcq_pairs.append({
                    'question': new_q,
                    'answer': new_a,
                    'type': 'mcq',
                    'contexts': pair.get('contexts', []),
                    'metadata': pair.get('metadata', {})
                })
    
    return factoid_pairs, list_pairs, mcq_pairs


def save_outputs(
    factoid_pairs: List[Dict],
    list_pairs: List[Dict],
    mcq_pairs: List[Dict],
    output_dir: str,
    num_questions: int
):
    """Save the final QA pairs to TSV files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combine and balance to exactly num_questions
    all_pairs = []
    
    # Calculate how many of each type to include
    min_per_type = 20
    total_available = len(factoid_pairs) + len(list_pairs) + len(mcq_pairs)
    
    if total_available >= num_questions:
        # Distribute evenly but ensure minimums
        remaining = num_questions - (min_per_type * 3)
        extra_per_type = remaining // 3
        
        take_factoid = min(min_per_type + extra_per_type, len(factoid_pairs))
        take_list = min(min_per_type + extra_per_type, len(list_pairs))
        take_mcq = min(min_per_type + extra_per_type, len(mcq_pairs))
        
        # Adjust if any type is short
        total_take = take_factoid + take_list + take_mcq
        if total_take < num_questions:
            # Add more from whichever has extra
            diff = num_questions - total_take
            if len(factoid_pairs) > take_factoid:
                take_factoid += min(diff, len(factoid_pairs) - take_factoid)
    else:
        take_factoid = len(factoid_pairs)
        take_list = len(list_pairs)
        take_mcq = len(mcq_pairs)
    
    all_pairs.extend(factoid_pairs[:take_factoid])
    all_pairs.extend(list_pairs[:take_list])
    all_pairs.extend(mcq_pairs[:take_mcq])
    
    # Shuffle for variety
    random.shuffle(all_pairs)
    
    # Save question.tsv
    question_file = output_path / "question.tsv"
    with open(question_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            # Clean question text
            q_text = pair['question'].replace('\t', ' ').replace('\n', ' ')
            f.write(f"{q_text}\t{pair['type']}\n")
    
    # Save answer.tsv
    answer_file = output_path / "answer.tsv"
    with open(answer_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            # Clean answer text
            a_text = pair['answer'].replace('\t', ' ').replace('\n', ' ')
            f.write(f"{a_text}\n")
    
    # Save evidence.tsv
    evidence_file = output_path / "evidence.tsv"
    with open(evidence_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            metadata = pair.get('metadata', {})
            filename = metadata.get('filename', 'unknown.txt')
            pdf_name = metadata.get('pdf_name', filename.split('_')[0] + '_' + filename.split('_')[1] if '_' in filename else 'unknown')
            f.write(f"{pdf_name}\t{filename}\n")
    
    # Print summary
    type_counts = {'factoid': 0, 'list': 0, 'mcq': 0}
    for pair in all_pairs:
        type_counts[pair['type']] = type_counts.get(pair['type'], 0) + 1
    
    print(f"\n{'='*60}")
    print("FINAL OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(all_pairs)}")
    print(f"  Factoid: {type_counts.get('factoid', 0)}")
    print(f"  List:    {type_counts.get('list', 0)}")
    print(f"  MCQ:     {type_counts.get('mcq', 0)}")
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - question.tsv")
    print(f"  - answer.tsv")
    print(f"  - evidence.tsv")
    
    # Validate
    min_check = min(type_counts.values())
    if min_check < 20:
        print(f"\n⚠️  WARNING: Some question types have fewer than 20 questions!")
        print(f"   Minimum required: 20, actual minimum: {min_check}")
    else:
        print(f"\n✓ All question types have 20+ questions. Ready for evaluation!")
    
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description='Generate QA pairs using RAGAS')
    parser.add_argument('--corpus_dir', type=str, default='data/corpus',
                        help='Directory containing text chunks')
    parser.add_argument('--output_dir', type=str, default='data/qa',
                        help='Directory to save QA files')
    parser.add_argument('--num_questions', type=int, default=100,
                        help='Total number of questions to generate')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--api_base', type=str, default=None,
                        help='OpenAI API base URL (for CMU Gateway)')
    
    args = parser.parse_args()
    
    # Get API credentials
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    api_base = args.api_base or os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
    
    if not api_key:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or use --api_key")
        sys.exit(1)
    
    print(f"{'='*60}")
    print("RAGAS-based Synthetic QA Generation")
    print(f"{'='*60}")
    print(f"Corpus directory: {args.corpus_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target questions: {args.num_questions}")
    print(f"API base: {api_base}")
    print(f"{'='*60}\n")
    
    # Step 1: Load corpus
    print("Step 1: Loading corpus documents...")
    documents = load_corpus(args.corpus_dir)
    
    if not documents:
        print("ERROR: No documents found in corpus directory")
        sys.exit(1)
    
    # Step 2: Generate raw QA pairs
    print("\nStep 2: Generating raw QA pairs...")
    
    # Try RAGAS first (works on Colab), fallback to Direct LLM (for local Windows where PyTorch is broken)
    if RAGAS_AVAILABLE:
        print("Using RAGAS TestsetGenerator (High Quality)")
        raw_pairs = generate_with_ragas(documents, args.num_questions, api_key, api_base)
    else:
        print("RAGAS unavailable - Using Direct LLM Generation (gpt-5)")
        raw_pairs = generate_with_llm_direct(documents, args.num_questions, api_key, api_base)
    
    if not raw_pairs:
        print("ERROR: No QA pairs generated")
        sys.exit(1)
    
    # Step 3: Process and transform to required types
    print("\nStep 3: Processing into factoid/list/mcq types...")
    factoid_pairs, list_pairs, mcq_pairs = process_qa_pairs(
        raw_pairs, args.num_questions, api_key, api_base
    )
    
    # Step 4: Save outputs
    print("\nStep 4: Saving outputs...")
    save_outputs(factoid_pairs, list_pairs, mcq_pairs, args.output_dir, args.num_questions)
    
    print("\n✓ QA generation complete!")


if __name__ == "__main__":
    main()
