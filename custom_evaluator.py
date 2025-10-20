"""
Custom Evaluation System for RAG Baseline

This module provides evaluation using the specific CSV format with multi-file and single-file queries,
and produces JSON output with the requested format.
"""

import pandas as pd
import numpy as np
import json
import ast
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from sklearn.metrics import ndcg_score

from rag_system import LLMSystemsRAG, QueryResult

logger = logging.getLogger(__name__)

@dataclass
class QueryEvaluationResult:
    """Results from evaluating a single query"""
    query: str
    true_filenames: List[str]
    retrieved_documents: List[Dict]  # List of {document_id, title, content, score}
    semantic_scores: List[float]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    is_relevant: List[bool]

class CustomRAGEvaluator:
    """
    Custom evaluator for the specific CSV format and JSON output requirements
    """
    
    def __init__(self, rag_system: LLMSystemsRAG):
        self.rag_system = rag_system
        self.k_values = [1, 3, 5, 10]
        
    def parse_filename_list(self, filename_str: str) -> List[str]:
        """Parse filename string from CSV (handles both single and multi-file formats)"""
        try:
            # Try to parse as Python list literal
            parsed = ast.literal_eval(filename_str)
            if isinstance(parsed, list):
                return [f.replace('.txt', '') for f in parsed]
            else:
                return [parsed.replace('.txt', '')]
        except:
            # Fallback: treat as single filename
            return [filename_str.replace('.txt', '')]
    
    def evaluate_single_file_query(self, query: str, true_filename: str, top_k: int = 10) -> QueryEvaluationResult:
        """Evaluate a single-file query"""
        # Retrieve documents
        results = self.rag_system.search(query, top_k=top_k)
        
        # Extract information
        retrieved_documents = []
        semantic_scores = []
        retrieved_doc_ids = []
        
        for result in results:
            retrieved_documents.append({
                "document_id": result.document_id,
                "title": result.metadata['title'],
                "content": result.content,
                "score": result.score
            })
            semantic_scores.append(result.score)
            retrieved_doc_ids.append(result.document_id)
        
        # Check relevance
        true_filename_clean = true_filename.replace('.txt', '')
        is_relevant = [doc_id == true_filename_clean for doc_id in retrieved_doc_ids]
        
        # Calculate metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            if k <= len(retrieved_doc_ids):
                # Precision@k
                precision_at_k[k] = sum(is_relevant[:k]) / k
                
                # Recall@k
                recall_at_k[k] = sum(is_relevant[:k]) / 1  # Only one relevant doc
                
                # NDCG@k
                relevance_scores = [1 if rel else 0 for rel in is_relevant[:k]]
                ideal_relevance = [1] + [0] * (k - 1)
                if len(relevance_scores) > 1:
                    ndcg_at_k[k] = ndcg_score([ideal_relevance], [relevance_scores])
                elif sum(relevance_scores) > 0:
                    ndcg_at_k[k] = 1.0  # Perfect score when only one relevant doc is retrieved
                else:
                    ndcg_at_k[k] = 0.0
            else:
                precision_at_k[k] = 0.0
                recall_at_k[k] = 0.0
                ndcg_at_k[k] = 0.0
        
        # MRR
        mrr = 0.0
        for i, rel in enumerate(is_relevant):
            if rel:
                mrr = 1.0 / (i + 1)
                break
        
        return QueryEvaluationResult(
            query=query,
            true_filenames=[true_filename_clean],
            retrieved_documents=retrieved_documents,
            semantic_scores=semantic_scores,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            is_relevant=is_relevant
        )
    
    def evaluate_multi_file_query(self, query: str, true_filenames: List[str], top_k: int = 10) -> QueryEvaluationResult:
        """Evaluate a multi-file query"""
        # Retrieve documents
        results = self.rag_system.search(query, top_k=top_k)
        
        # Extract information
        retrieved_documents = []
        semantic_scores = []
        retrieved_doc_ids = []
        
        for result in results:
            retrieved_documents.append({
                "document_id": result.document_id,
                "title": result.metadata['title'],
                "content": result.content,
                "score": result.score
            })
            semantic_scores.append(result.score)
            retrieved_doc_ids.append(result.document_id)
        
        # Check relevance
        true_filenames_clean = [f.replace('.txt', '') for f in true_filenames]
        is_relevant = [doc_id in true_filenames_clean for doc_id in retrieved_doc_ids]
        
        # Calculate metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            if k <= len(retrieved_doc_ids):
                # Precision@k
                precision_at_k[k] = sum(is_relevant[:k]) / k
                
                # Recall@k
                recall_at_k[k] = sum(is_relevant[:k]) / len(true_filenames_clean)
                
                # NDCG@k
                relevance_scores = [1 if rel else 0 for rel in is_relevant[:k]]
                ideal_relevance = [1] * min(k, len(true_filenames_clean)) + [0] * max(0, k - len(true_filenames_clean))
                if len(relevance_scores) > 1:
                    ndcg_at_k[k] = ndcg_score([ideal_relevance], [relevance_scores])
                elif sum(relevance_scores) > 0:
                    ndcg_at_k[k] = 1.0  # Perfect score when relevant docs are retrieved
                else:
                    ndcg_at_k[k] = 0.0
            else:
                precision_at_k[k] = 0.0
                recall_at_k[k] = 0.0
                ndcg_at_k[k] = 0.0
        
        # MRR
        mrr = 0.0
        for i, rel in enumerate(is_relevant):
            if rel:
                mrr = 1.0 / (i + 1)
                break
        
        return QueryEvaluationResult(
            query=query,
            true_filenames=true_filenames_clean,
            retrieved_documents=retrieved_documents,
            semantic_scores=semantic_scores,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            is_relevant=is_relevant
        )
    
    def evaluate_from_csv(self, single_file_csv: str, multi_file_csv: str, top_k: int = 10) -> Dict:
        """Evaluate system using both CSV files and return JSON results"""
        
        # Load single file queries
        single_df = pd.read_csv(single_file_csv)
        logger.info(f"Loaded {len(single_df)} single-file queries")
        
        # Load multi file queries
        multi_df = pd.read_csv(multi_file_csv)
        logger.info(f"Loaded {len(multi_df)} multi-file queries")
        
        all_results = []
        
        # Evaluate single file queries
        for _, row in single_df.iterrows():
            if pd.isna(row['query']) or pd.isna(row['filename']):
                continue
                
            logger.info(f"Evaluating single-file query: {row['query'][:50]}...")
            result = self.evaluate_single_file_query(
                query=row['query'],
                true_filename=row['filename'],
                top_k=top_k
            )
            all_results.append(result)
        
        # Evaluate multi file queries
        for _, row in multi_df.iterrows():
            if pd.isna(row['query']) or pd.isna(row['filename']):
                continue
                
            logger.info(f"Evaluating multi-file query: {row['query'][:50]}...")
            true_filenames = self.parse_filename_list(row['filename'])
            result = self.evaluate_multi_file_query(
                query=row['query'],
                true_filenames=true_filenames,
                top_k=top_k
            )
            all_results.append(result)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_results)
        
        # Create JSON output
        json_output = {
            "evaluation_summary": {
                "total_queries": len(all_results),
                "single_file_queries": len(single_df),
                "multi_file_queries": len(multi_df),
                "overall_metrics": overall_metrics
            },
            "query_results": []
        }
        
        # Add individual query results
        for result in all_results:
            query_result = {
                "query": result.query,
                "true_filenames": result.true_filenames,
                "retrieved_documents": result.retrieved_documents,
                "semantic_scores": result.semantic_scores,
                "metrics": {
                    "precision_at_k": result.precision_at_k,
                    "recall_at_k": result.recall_at_k,
                    "mrr": result.mrr,
                    "ndcg_at_k": result.ndcg_at_k
                },
                "is_relevant": result.is_relevant
            }
            json_output["query_results"].append(query_result)
        
        return json_output
    
    def _calculate_overall_metrics(self, results: List[QueryEvaluationResult]) -> Dict:
        """Calculate overall metrics across all queries"""
        overall_precision_at_k = {}
        overall_recall_at_k = {}
        overall_ndcg_at_k = {}
        
        for k in self.k_values:
            precisions = [r.precision_at_k.get(k, 0) for r in results if k in r.precision_at_k]
            recalls = [r.recall_at_k.get(k, 0) for r in results if k in r.recall_at_k]
            ndcgs = [r.ndcg_at_k.get(k, 0) for r in results if k in r.ndcg_at_k]
            
            overall_precision_at_k[k] = np.mean(precisions) if precisions else 0
            overall_recall_at_k[k] = np.mean(recalls) if recalls else 0
            overall_ndcg_at_k[k] = np.mean(ndcgs) if ndcgs else 0
        
        overall_mrr = np.mean([r.mrr for r in results])
        
        return {
            "precision_at_k": overall_precision_at_k,
            "recall_at_k": overall_recall_at_k,
            "mrr": overall_mrr,
            "ndcg_at_k": overall_ndcg_at_k
        }

def main():
    """Main function to run custom evaluation"""
    # Initialize RAG system with smaller chunk size
    rag = LLMSystemsRAG(
        chunk_size=256,  # Smaller chunks for better retrieval
        chunk_overlap=25
    )
    
    # Build vector index
    rag.build_vector_index()
    
    # Initialize evaluator
    evaluator = CustomRAGEvaluator(rag)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate_from_csv(
        single_file_csv="queries/single_file_retrieval_queries.csv",
        multi_file_csv="queries/multi_file_retrieval_queries.csv",
        top_k=10
    )
    
    # Save results
    output_file = "custom_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation completed. Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("CUSTOM RAG EVALUATION RESULTS")
    print("="*80)
    
    summary = results["evaluation_summary"]
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Single-file Queries: {summary['single_file_queries']}")
    print(f"Multi-file Queries: {summary['multi_file_queries']}")
    
    metrics = summary["overall_metrics"]
    print(f"\nOverall MRR: {metrics['mrr']:.4f}")
    
    print("\nPrecision@K:")
    for k, p in metrics['precision_at_k'].items():
        print(f"  P@{k}: {p:.4f}")
    
    print("\nRecall@K:")
    for k, r in metrics['recall_at_k'].items():
        print(f"  R@{k}: {r:.4f}")
    
    print("\nNDCG@K:")
    for k, n in metrics['ndcg_at_k'].items():
        print(f"  NDCG@{k}: {n:.4f}")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
