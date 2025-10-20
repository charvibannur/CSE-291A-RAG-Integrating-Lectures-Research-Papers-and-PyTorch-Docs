# RAG Custom Evaluator

A minimal RAG (Retrieval-Augmented Generation) evaluation system for Large Language Model Systems domain.

## Overview

This system evaluates RAG performance using custom CSV queries with both single-file and multi-file retrieval scenarios. It produces comprehensive JSON output with retrieval metrics including precision@k, recall@k, MRR, and NDCG@k.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data (Required)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 3. Run Evaluation

```bash
python custom_evaluator.py
```

## What It Does

- **Loads Documents**: Processes 105+ text documents from `text_docs/` directory
- **Builds Vector Index**: Creates ChromaDB vector database with sentence-transformer embeddings
- **Evaluates Queries**: Tests against single-file and multi-file retrieval scenarios
- **Generates Metrics**: Computes precision@k, recall@k, MRR, NDCG@k for each query
- **Outputs JSON**: Creates `custom_evaluation_results.json` with detailed results

## File Structure

```
├── custom_evaluator.py          # Main evaluation script
├── rag_system.py               # RAG system implementation
├── requirements.txt             # Python dependencies
├── queries/                    # Evaluation queries
│   ├── single_file_retrieval_queries.csv
│   └── multi_file_retrieval_queries.csv
├── text_docs/                  # Document corpus (105+ files)
└── README.md                   # This file
```

## Output Format

The system generates `custom_evaluation_results.json` with:

```json
{
  "evaluation_summary": {
    "total_queries": 15,
    "single_file_queries": 10,
    "multi_file_queries": 5,
    "overall_metrics": {
      "precision_at_k": {"1": 0.6, "3": 0.7, "5": 0.8},
      "recall_at_k": {"1": 0.4, "3": 0.6, "5": 0.8},
      "mrr": 0.65,
      "ndcg_at_k": {"1": 0.6, "3": 0.7, "5": 0.8}
    }
  },
  "query_results": [
    {
      "query": "What is FlashAttention?",
      "true_filenames": ["23_flashattention_2205_14135"],
      "retrieved_documents": [...],
      "semantic_scores": [0.8, 0.7, 0.6],
      "metrics": {...},
      "is_relevant": [true, false, false]
    }
  ]
}
```

## Key Features

- **Semantic Search**: Uses sentence-transformers for vector similarity
- **Flexible Queries**: Supports both single and multi-file retrieval scenarios
- **Comprehensive Metrics**: Precision@k, Recall@k, MRR, NDCG@k
- **JSON Output**: Machine-readable results for further analysis
- **Minimal Dependencies**: Only essential packages required

## Troubleshooting

- **Memory Issues**: Reduce chunk size in `rag_system.py` if needed
- **Slow Performance**: The first run builds the vector index (may take a few minutes)
- **Missing Files**: Ensure `text_docs/` and `queries/` directories exist

## Domain

This evaluator is specifically designed for the **Large Language Model Systems** domain, including:
- Research papers on neural networks and transformers
- CMU lecture materials on LLM systems
- PyTorch documentation
- Recent research on LLM optimization techniques