#!/usr/bin/env python3
"""
RAG Evaluation using DeepEval LLM-as-a-Judge Metrics

This script evaluates RAG systems using 5 key metrics:
1. ContextualPrecisionMetric: Reranker ranks relevant nodes higher
2. ContextualRecallMetric: Embedding model captures relevant information
3. ContextualRelevancyMetric: Chunk size/top-K retrieves without irrelevancies
4. AnswerRelevancyMetric: Generator outputs relevant/helpful responses
5. FaithfulnessMetric: Generator doesn't hallucinate or contradict context
"""

import os
import sys
import json
import ast
import pandas as pd
import warnings
from typing import List, Dict
from tqdm import tqdm
import re
from pathlib import Path

# Add paths to sys.path
sys.path.append(str(Path(__file__).parent))

# Import DeepEval
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

# Import RAG models
try:
    from baselines.baseline2.rag_semanticsearch import LLMSystemsRAG as Baseline2RAG
    from model.hybrid_rag import LLMSystemsRAG as HybridRAG
except ImportError as e:
    print(f"Failed to import models: {e}")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Baseline 1 (BM25) Implementation ---
import math
from collections import Counter

class Baseline1BM25:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.pdfs = {}
        self.index = {}
        self.per_word = {}
        self.avg_val = 0
        self.k1 = 1.5
        self.b = 0.75
        self._load_data()
        self._build_index()

    def _load_data(self):
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        pdf = f.read()
                        pdf = re.sub(r'\n\s*\n', '\n', pdf)
                        pdf = re.sub(r'\s+', ' ', pdf)
                        self.pdfs[file_path] = pdf

    def _build_index(self):
        total_length = 0
        for path_file, pdf_text_val in self.pdfs.items():
            pdf_text_val = pdf_text_val.lower()
            words = re.findall(r'\w+', pdf_text_val)
            total_length += len(words)
            self.per_word[path_file] = len(words)
            word_freq = Counter(words)
            for word, freq in word_freq.items():
                if word not in self.index:
                    self.index[word] = {}
                self.index[word][path_file] = freq
        
        if len(self.pdfs) > 0:
            self.avg_val = total_length / len(self.pdfs)

    def _bm25_score(self, term, path_file):
        if term not in self.index or path_file not in self.index[term]:
            return 0
        freq_u = self.index[term][path_file]
        size = self.per_word[path_file]
        normalized_length = size / self.avg_val
        N = len(self.per_word)
        term_total_ = len(self.index[term])
        indx_freq = math.log((N - term_total_ + 0.5) / (term_total_ + 0.5) + 1)
        numr = freq_u * (self.k1 + 1)
        denom = freq_u + self.k1 * (1 - self.b + self.b * normalized_length)
        return indx_freq * numr / denom

    def search(self, query, top_k=3):
        query = query.lower()
        query_terms = re.findall(r'\w+', query)
        doc_scores = {}
        for term in query_terms:
            if term in self.index:
                for path_file in self.index[term]:
                    score = self._bm25_score(term, path_file)
                    if path_file not in doc_scores:
                        doc_scores[path_file] = 0.0
                    doc_scores[path_file] += score

        scored_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for path, score in scored_docs:
            if score > 0:
                # Return first 2000 chars as snippet
                results.append(self.pdfs[path][:2000])
        return results


# --- LLM Answer Generation ---
from openai import OpenAI as OpenAIClient

client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))

# High-quality prompt template for RAG generation
RAG_PROMPT_TEMPLATE = """You are an expert AI assistant specializing in Large Language Model Systems and Deep Learning.

Your task is to provide a comprehensive, accurate, and well-structured answer to the user's question based ONLY on the provided context documents.

## Instructions:
1. Carefully read and analyze ALL provided context documents
2. Synthesize information from multiple sources when relevant
3. Provide specific technical details, numbers, and examples when available
4. Structure your answer clearly with proper formatting
5. If the context contains conflicting information, acknowledge it
6. If the context does not contain sufficient information to answer the question, clearly state: "Based on the provided documents, I cannot fully answer this question because [reason]."

## Context Documents:
{context}

## User Question:
{question}

## Your Answer:
Provide a detailed, technically accurate response that directly addresses the question. Use bullet points or numbered lists for clarity when appropriate."""

def generate_answer_with_llm(query: str, context_list: List[str]) -> str:
    """Generate answer using GPT-4o with optimized RAG prompt."""
    context_str = "\n\n---\n\n".join([f"[Document {i+1}]:\n{ctx}" for i, ctx in enumerate(context_list)])
    prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=query)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI assistant. Provide accurate, well-structured answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"


# --- Main Evaluation ---
def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY is required.")
        sys.exit(1)

    # Initialize 5 DeepEval metrics with GPT-4o-mini as judge
    judge_model = "gpt-4o"
    
    metrics = [
        # Retrieval metrics (evaluate retriever quality)
        ContextualPrecisionMetric(threshold=0.25, model=judge_model),  # Reranker quality
        ContextualRecallMetric(threshold=0.35, model=judge_model),     # Embedding model quality
        ContextualRelevancyMetric(threshold=0.25, model=judge_model),  # Chunk size/top-K quality
        # Generation metrics (evaluate generator quality)
        AnswerRelevancyMetric(threshold=0.35, model=judge_model),      # Prompt template quality
        FaithfulnessMetric(threshold=0.35, model=judge_model),         # LLM hallucination check
    ]

    # Load queries from both files
    multi_file_path = "queries/multi_file_retrieval_queries.csv"
    single_file_path = "queries/single_file_retrieval_queries.csv"
    
    dfs = []
    
    if os.path.exists(multi_file_path):
        df_multi = pd.read_csv(multi_file_path)
        # Standardize column name
        if 'gold-text' in df_multi.columns:
            df_multi = df_multi.rename(columns={'gold-text': 'expected_output'})
        df_multi['query_type'] = 'multi_file'
        dfs.append(df_multi)
        print(f"Loaded {len(df_multi)} queries from multi_file_retrieval_queries.csv")
    
    if os.path.exists(single_file_path):
        df_single = pd.read_csv(single_file_path)
        # Standardize column name (single file uses 'gold-doc')
        if 'gold-doc' in df_single.columns:
            df_single = df_single.rename(columns={'gold-doc': 'expected_output'})
        elif 'gold-text' in df_single.columns:
            df_single = df_single.rename(columns={'gold-text': 'expected_output'})
        df_single['query_type'] = 'single_file'
        dfs.append(df_single)
        print(f"Loaded {len(df_single)} queries from single_file_retrieval_queries.csv")
    
    if not dfs:
        print("No query files found!")
        sys.exit(1)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total queries: {len(df)}")
    
    # Initialize RAG models
    print("Initializing RAG models...")
    b1 = Baseline1BM25("data")
    
    b2 = Baseline2RAG(data_dir="data", vector_db_path="vector_db_baseline2")
    if b2.collection.count() == 0:
        print("Building Baseline 2 index...")
        b2.build_vector_index()
    
    hybrid = HybridRAG(data_dir="data", vector_db_path="vector_db_hybrid")
    if hybrid.collection.count() == 0:
        print("Building Hybrid index...")
        hybrid.build_index()

    models = {
        "Baseline 1 (BM25)": b1,
        "Baseline 2 (Semantic)": b2,
        "Hybrid RAG": hybrid
    }
    
    results_summary = []
    metric_names = [m.__class__.__name__ for m in metrics]

    print(f"\nEvaluating {len(df)} queries across {len(models)} models...")
    print(f"Metrics: {', '.join(metric_names)}")
    print(f"Total evaluations: {len(df) * len(models)} (queries × models)\n")
    
    total_evals = len(df) * len(models)
    eval_count = 0
    
    for query_idx, row in enumerate(df.iterrows()):
        _, row = row
        query = row['query']
        gold_text_raw = row['expected_output']
        
        # Parse gold text (may be a tuple string or plain text)
        try:
            expected_output_list = list(ast.literal_eval(gold_text_raw))
            expected_output = "\n".join(expected_output_list)
        except:
            expected_output = str(gold_text_raw)
        
        for model_idx, (model_name, model_obj) in enumerate(models.items()):
            eval_count += 1
            # Progress display
            print(f"\r[{eval_count}/{total_evals}] Query {query_idx+1}/{len(df)} | Model: {model_name[:20]:<20} | ", end="", flush=True)
            
            # 1. Retrieve context
            print("Retrieving...", end="", flush=True)
            if model_name == "Baseline 1 (BM25)":
                retrieval_context = model_obj.search(query, top_k=3)
            elif model_name == "Baseline 2 (Semantic)":
                res = model_obj.search(query, top_k=3)
                retrieval_context = [r.content for r in res]
            else:  # Hybrid
                res = model_obj.hybrid_search(query, top_k=3)
                retrieval_context = [r.content for r in res]

            # 2. Generate answer using LLM
            print(" Generating...", end="", flush=True)
            actual_output = generate_answer_with_llm(query, retrieval_context)

            # 3. Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=expected_output
            )

            # 4. Evaluate with all metrics
            print(" Evaluating metrics...", end="", flush=True)
            metrics_results = {"model": model_name, "query": query, "actual_output": actual_output}
            
            for metric in metrics:
                try:
                    metric.measure(test_case)
                    metrics_results[metric.__class__.__name__] = metric.score
                except Exception as e:
                    print(f"\n  Error measuring {metric.__class__.__name__}: {e}")
                    metrics_results[metric.__class__.__name__] = None

            results_summary.append(metrics_results)
            print(" Done!", end="", flush=True)
    
    print("\n")  # New line after progress

    # Save results
    results_df = pd.DataFrame(results_summary)
    output_path = "results/llm_judge_evaluation_results.csv"
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*100)
    print("EVALUATION SUMMARY - LLM-as-a-Judge Metrics")
    print("="*100)
    
    # Aggregated metrics by model
    print("\n### Aggregated Metrics (Mean) by Model ###\n")
    agg = results_df.groupby("model")[metric_names].mean()
    print(agg.to_string())
    
    # Per-query results
    print("\n\n### Per-Query Results (Sample) ###\n")
    display_cols = ["model", "query"] + metric_names
    print(results_df[display_cols].head(9).to_string())
    
    print(f"\n\nFull results saved to: {output_path}")
    
    # Save summary JSON
    summary = {
        "total_queries": len(df),
        "models_evaluated": list(models.keys()),
        "metrics_used": metric_names,
        "aggregated_results": agg.to_dict()
    }
    with open("results/llm_judge_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to: results/llm_judge_evaluation_summary.json")


if __name__ == "__main__":
    main()
