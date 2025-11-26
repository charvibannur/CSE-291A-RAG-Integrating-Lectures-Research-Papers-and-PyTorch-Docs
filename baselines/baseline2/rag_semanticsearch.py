"""
RAG Baseline System for Large Language Model Systems Domain

This module implements a comprehensive RAG (Retrieval-Augmented Generation) system
for the LLM Systems domain, including:
- Improved Document chunking (Recursive Character Splitter)
- Hybrid Search (Vector + BM25)
- Reranking (Cross-Encoder)
- Vector database integration with ChromaDB
- Evaluation pipeline
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from tqdm import tqdm
import re
import csv

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document with metadata"""
    id: str
    content: str
    title: str
    source: str
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None

@dataclass
class QueryResult:
    """Represents a query result with relevance score"""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict
    rank: int = 0

class RecursiveCharacterTextSplitter:
    """
    Splits text by recursively looking at characters.
    Recursively tries to split by different characters to find one
    that works.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        final_chunks = []
        separator = self.separators[-1]
        
        # Find the best separator
        for sep in self.separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
                
        # Split
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text) # Split by character
            
        # Merge splits
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            if current_length + split_len + (len(separator) if current_chunk else 0) > self.chunk_size:
                if current_chunk:
                    doc_chunk = separator.join(current_chunk)
                    if doc_chunk.strip():
                        final_chunks.append(doc_chunk)
                    
                    # Handle overlap
                    if self.chunk_overlap > 0:
                        # Keep some splits for overlap
                        overlap_len = 0
                        new_chunk = []
                        for s in reversed(current_chunk):
                            if overlap_len + len(s) < self.chunk_overlap:
                                new_chunk.insert(0, s)
                                overlap_len += len(s)
                            else:
                                break
                        current_chunk = new_chunk
                        current_length = overlap_len
                    else:
                        current_chunk = []
                        current_length = 0
                        
            current_chunk.append(split)
            current_length += split_len + (len(separator) if len(current_chunk) > 1 else 0)
            
        if current_chunk:
            doc_chunk = separator.join(current_chunk)
            if doc_chunk.strip():
                final_chunks.append(doc_chunk)
                
        return final_chunks

class LLMSystemsRAG:
    """
    Advanced RAG system for Large Language Model Systems domain
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 vector_db_path: str = "vector_db",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 100):
        """
        Initialize the RAG system
        """
        self.data_dir = Path(data_dir)
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize reranking model
        logger.info(f"Loading reranking model: {rerank_model}")
        self.reranker = CrossEncoder(rerank_model)
        
        # Initialize ChromaDB
        self._init_vector_db()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_mapping = [] # Maps BM25 index to (doc_id, chunk_index)
        
    def _init_vector_db(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="llm_systems_docs_v2",
            metadata={"description": "LLM Systems domain documents - Phase 2"}
        )
        
        logger.info(f"Vector database initialized at {self.vector_db_path}")
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the data directory"""
        documents = []
        
        logger.info(f"Loading documents from {self.data_dir}")
        
        for file_path in tqdm(list(self.data_dir.glob("*.txt")), desc="Loading documents"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if len(content) < 100:  # Skip very short files
                    continue
                
                # Extract title from filename or first line
                title = self._extract_title(file_path.stem, content)
                
                doc = Document(
                    id=file_path.stem,
                    content=content,
                    title=title,
                    source=str(file_path)
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _extract_title(self, filename: str, content: str) -> str:
        """Extract title from filename or document content"""
        # Try to extract from first few lines
        lines = content.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                line = re.sub(r'^\d+\s*', '', line)
                line = re.sub(r'[^\w\s\-\.]', '', line)
                if line:
                    return line
        return filename.replace('_', ' ').title()
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into overlapping chunks using recursive splitter"""
        chunks = []
        split_texts = self.text_splitter.split_text(document.content)
        
        for i, text in enumerate(split_texts):
            chunk_doc = Document(
                id=f"{document.id}_chunk_{i}",
                content=text,
                title=document.title,
                source=document.source,
                chunk_id=f"{document.id}_chunk_{i}",
                chunk_index=i
            )
            chunks.append(chunk_doc)
            
        return chunks
    
    def build_index(self, documents: Optional[List[Document]] = None):
        """Build both Vector and BM25 indices"""
        if documents is None:
            documents = self.load_documents()
        
        logger.info("Building indices...")
        
        # Reset vector DB
        if self.collection.count() > 0:
            self.client.delete_collection("llm_systems_docs_v2")
            self.collection = self.client.get_or_create_collection(
                name="llm_systems_docs_v2",
                metadata={"description": "LLM Systems domain documents - Phase 2"}
            )
        
        all_chunks = []
        chunk_metadata = []
        chunk_ids = []
        
        # For BM25
        tokenized_corpus = []
        self.bm25_mapping = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            chunks = self.chunk_document(doc)
            
            for chunk in chunks:
                # Vector DB data
                all_chunks.append(chunk.content)
                chunk_metadata.append({
                    "document_id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "chunk_index": chunk.chunk_index
                })
                chunk_ids.append(chunk.chunk_id)
                
                # BM25 data
                tokens = word_tokenize(chunk.content.lower())
                tokenized_corpus.append(tokens)
                self.bm25_mapping.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk_metadata[-1]
                })
        
        # Build BM25 Index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = tokenized_corpus # Keep if needed, though BM25 stores stats
        
        # Build Vector Index
        logger.info("Generating embeddings and populating Vector DB...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        batch_size = 1000
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Adding to Vector DB", total=total_batches):
            end_idx = min(i + batch_size, len(all_chunks))
            self.collection.add(
                embeddings=embeddings[i:end_idx].tolist(),
                documents=all_chunks[i:end_idx],
                metadatas=chunk_metadata[i:end_idx],
                ids=chunk_ids[i:end_idx]
            )
            
        logger.info(f"Indices built with {len(all_chunks)} chunks")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """
        Perform hybrid search (Vector + BM25) followed by Reranking
        """
        # 1. Vector Search
        query_embedding = self.embedding_model.encode([query])
        vector_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k * 2 # Fetch more for reranking
        )
        
        vector_candidates = {}
        if vector_results['ids']:
            for i, chunk_id in enumerate(vector_results['ids'][0]):
                vector_candidates[chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': vector_results['documents'][0][i],
                    'metadata': vector_results['metadatas'][0][i],
                    'score': vector_results['distances'][0][i] # Lower is better for distance
                }

        # 2. BM25 Search
        if self.bm25:
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_bm25 = np.argsort(bm25_scores)[::-1][:top_k * 2]
            
            bm25_candidates = {}
            for idx in top_n_bm25:
                item = self.bm25_mapping[idx]
                bm25_candidates[item['chunk_id']] = {
                    'chunk_id': item['chunk_id'],
                    'content': item['content'],
                    'metadata': item['metadata'],
                    'score': bm25_scores[idx] # Higher is better
                }
        else:
            bm25_candidates = {}

        # 3. Combine Candidates (Union)
        all_candidates = {**vector_candidates, **bm25_candidates}
        unique_candidates = list(all_candidates.values())
        
        if not unique_candidates:
            return []

        # 4. Rerank
        cross_inp = [[query, cand['content']] for cand in unique_candidates]
        cross_scores = self.reranker.predict(cross_inp)
        
        for i, cand in enumerate(unique_candidates):
            cand['rerank_score'] = cross_scores[i]
            
        # Sort by rerank score (descending)
        ranked_results = sorted(unique_candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Format output
        final_results = []
        for i, res in enumerate(ranked_results[:top_k]):
            final_results.append(QueryResult(
                document_id=res['metadata']['document_id'],
                chunk_id=res['chunk_id'],
                content=res['content'],
                score=float(res['rerank_score']),
                metadata=res['metadata'],
                rank=i+1
            ))
            
        return final_results

    def evaluate_queries(self, queries_file: str, output_file: str):
        """
        Run retrieval for queries in a CSV and save results to JSON
        """
        logger.info(f"Evaluating queries from {queries_file}")
        df = pd.read_csv(queries_file)
        
        # Detect query column
        query_col = None
        for col in ['query', 'question', 'Query', 'Question']:
            if col in df.columns:
                query_col = col
                break
        
        if not query_col:
            raise ValueError(f"No query column found in {queries_file}")
            
        results_data = {"queries": []}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            query = row[query_col]
            results = self.hybrid_search(query, top_k=10)
            
            query_results = []
            for res in results:
                query_results.append({
                    "doc": res.document_id,
                    "filename": res.document_id, # Evaluator expects this
                    "text": res.content,
                    "score": res.score,
                    "rank": res.rank
                })
                
            results_data["queries"].append({
                "query": query,
                "results": query_results
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")

    def generate_answer(self, query: str) -> str:
        """
        Generate an answer for the query using retrieved context.
        (Placeholder for generation pipeline)
        """
        results = self.hybrid_search(query, top_k=3)
        context = "\n\n".join([r.content for r in results])
        
        # In a real system, you would pass this to an LLM.
        # Here we return a structured response demonstrating the context.
        return f"Based on the following context:\n\n{context}\n\n[Answer Generation Placeholder]"

def main():
    rag = LLMSystemsRAG()
    
    # Build index if needed (or force rebuild)
    rag.build_index()
    
    # Run Evaluation
    queries_dir = Path("queries")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Combine queries for evaluation
    multi_file = queries_dir / "multi_file_retrieval_queries.csv"
    single_file = queries_dir / "single_file_retrieval_queries.csv"
    
    # We can run evaluation on both and merge, or just one. 
    # The evaluator script expects a single JSON with all results.
    # Let's create a combined dataframe or just iterate both.
    
    all_queries = []
    if multi_file.exists():
        all_queries.append(pd.read_csv(multi_file))
    if single_file.exists():
        all_queries.append(pd.read_csv(single_file))
        
    if all_queries:
        combined_df = pd.concat(all_queries, ignore_index=True)
        # Save temporary combined csv
        combined_csv = results_dir / "combined_queries.csv"
        combined_df.to_csv(combined_csv, index=False)
        
        rag.evaluate_queries(str(combined_csv), str(results_dir / "rag_results.json"))
        
    # Interactive demo
    print("\n" + "="*80)
    print("RAG SYSTEM DEMO")
    print("="*80)
    
    demo_query = "What is FlashAttention?"
    print(f"Query: {demo_query}")
    results = rag.hybrid_search(demo_query, top_k=3)
    for r in results:
        print(f"\n[{r.score:.4f}] {r.metadata['title']}")
        print(f"{r.content[:200]}...")

if __name__ == "__main__":
    main()
