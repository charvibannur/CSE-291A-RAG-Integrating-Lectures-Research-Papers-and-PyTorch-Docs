#!/usr/bin/env python3
import os
import sys
import time
import json
import re
import math
import statistics
import logging
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Baseline 1 Implementation (BM25 from scratch) ---
class Baseline1BM25:
    def __init__(self, data_path):
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
        logger.info("[Baseline 1] Loading data...")
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
        logger.info("[Baseline 1] Building index...")
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
        else:
            self.avg_val = 0

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
        return scored_docs

# --- Import Baseline 2 and Hybrid ---
# Add paths to sys.path to allow imports
sys.path.append(str(Path(__file__).parent))
try:
    from baselines.baseline2.rag_semanticsearch import LLMSystemsRAG as Baseline2RAG
    from model.hybrid_rag import LLMSystemsRAG as HybridRAG
except ImportError as e:
    logger.error(f"Failed to import models: {e}")
    sys.exit(1)

def run_evaluation():
    results = {}
    data_dir = "data"
    queries_path = "queries/multi_file_retrieval_queries.csv" # Using multi file as primary test set
    
    if not os.path.exists(queries_path):
        logger.warning(f"Queries file not found at {queries_path}, trying single file...")
        queries_path = "queries/single_file_retrieval_queries.csv"
        
    df = pd.read_csv(queries_path)
    # Take a sample of queries for latency testing (e.g., 20)
    test_queries = df.sample(n=min(20, len(df)), random_state=42)
    
    query_col = None
    for col in ['query', 'question', 'Query', 'Question']:
        if col in df.columns:
            query_col = col
            break
    if not query_col:
        logger.error("Could not find query column")
        return

    queries_list = test_queries[query_col].tolist()
    logger.info(f"Using {len(queries_list)} queries for latency measurement")

    # --- 1. Measure Baseline 1 (BM25) ---
    logger.info("Initializing Baseline 1 (BM25)...")
    start_time = time.time()
    b1 = Baseline1BM25(data_dir)
    init_time_b1 = time.time() - start_time
    
    vis_latencies_b1 = []
    logger.info("Running Baseline 1...")
    for q in tqdm(queries_list):
        t0 = time.time()
        b1.search(q)
        vis_latencies_b1.append(time.time() - t0)
    
    results['Baseline 1 (BM25)'] = {
        'init_time': init_time_b1,
        'latencies': vis_latencies_b1
    }

    # --- 2. Measure Baseline 2 (Semantic Search) ---
    logger.info("Initializing Baseline 2 (Semantic Search)...")
    # Note: Baseline 2 creates a persistent DB. We might want to use a temp one or just measure load.
    # It reuses 'vector_db' by default. Let's use a separate path to avoid conflicts if possible/needed,
    # but the class defaults to 'vector_db'. Let's stick to defaults but be aware it might take time to build if not present.
    start_time = time.time()
    # Assuming documents are already in 'data'
    # The default data_dir in Baseline2 is 'text_docs', we need to override to 'data'
    b2 = Baseline2RAG(data_dir=data_dir, vector_db_path="vector_db_baseline2")
    # We need to ensure index is built. 
    # Check if we need to call build_vector_index(). 
    # The class loads embeddings. If we want to measure *inference* only, we should assume index specific stuff is done or included in init.
    # But usually build_vector_index is explicit.
    if b2.collection.count() == 0:
        logger.info("Building Baseline 2 Index...")
        b2.build_vector_index()
    init_time_b2 = time.time() - start_time

    vis_latencies_b2 = []
    logger.info("Running Baseline 2...")
    for q in tqdm(queries_list):
        t0 = time.time()
        b2.search(q)
        vis_latencies_b2.append(time.time() - t0)

    results['Baseline 2 (Semantic)'] = {
        'init_time': init_time_b2,
        'latencies': vis_latencies_b2
    }

    # --- 3. Measure Hybrid Model ---
    if "OPENAI_API_KEY" not in os.environ:
        logger.warning("OPENAI_API_KEY not found. Skipping Hybrid Model.")
        results['Hybrid RAG'] = {'init_time': 0, 'latencies': []}
    else:
        logger.info("Initializing Hybrid Model...")
        start_time = time.time()
        # Hybrid rag uses 'vector_db' by default too. Let's use 'vector_db_hybrid'
        hybrid = HybridRAG(data_dir=data_dir, vector_db_path="vector_db_hybrid")
        # Ensure index exists
        # In hybrid_rag.py, there is build_index()
        # We can check collection count
        if hybrid.collection.count() == 0:
             logger.info("Building Hybrid Index...")
             hybrid.build_index()
        
        init_time_hybrid = time.time() - start_time
        
        vis_latencies_hybrid = []
        logger.info("Running Hybrid Model...")
        for q in tqdm(queries_list):
            t0 = time.time()
            hybrid.hybrid_search(q)
            vis_latencies_hybrid.append(time.time() - t0)
            
        results['Hybrid RAG'] = {
            'init_time': init_time_hybrid,
            'latencies': vis_latencies_hybrid
        }

    # --- Report ---
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'Avg Latency (s)':<15} | {'Median (s)':<10} | {'P95 (s)':<10} | {'Init Time (s)':<15}")
    print("-" * 80)
    
    summary_data = []

    for model_name, data in results.items():
        if not data['latencies']:
            print(f"{model_name:<25} | {'N/A':<15} | {'N/A':<10} | {'N/A':<10} | {data.get('init_time', 0):.2f}")
            continue
            
        avg_lat = statistics.mean(data['latencies'])
        median_lat = statistics.median(data['latencies'])
        p95_lat = sorted(data['latencies'])[int(0.95 * len(data['latencies']))]
        init_t = data['init_time']
        
        print(f"{model_name:<25} | {avg_lat:.4f}          | {median_lat:.4f}     | {p95_lat:.4f}     | {init_t:.2f}")
        
        summary_data.append({
            "model": model_name,
            "avg_latency": avg_lat,
            "median_latency": median_lat,
            "p95_latency": p95_lat,
            "init_time": init_t
        })

    # Save results
    output_file = "results/latency_results.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    run_evaluation()
