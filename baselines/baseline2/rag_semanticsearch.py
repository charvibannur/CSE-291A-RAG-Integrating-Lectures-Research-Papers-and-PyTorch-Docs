"""
RAG Baseline System for Large Language Model Systems Domain

This module implements a comprehensive RAG (Retrieval-Augmented Generation) system
for the LLM Systems domain, including:
- Document chunking and embedding generation
- Vector database integration with ChromaDB
- Retrieval and ranking algorithms
- Evaluation metrics and benchmarking
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm
import re

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

class LLMSystemsRAG:
    """
    RAG system specifically designed for Large Language Model Systems domain
    """
    
    def __init__(self, 
                 data_dir: str = "text_docs",
                 vector_db_path: str = "vector_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize the RAG system
        
        Args:
            data_dir: Directory containing text documents
            vector_db_path: Path to store ChromaDB
            embedding_model: Sentence transformer model name
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self._init_vector_db()
        
        # Initialize text processor
        self.stop_words = set(stopwords.words('english'))
        
    def _init_vector_db(self):
        """Initialize ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="llm_systems_docs",
            metadata={"description": "LLM Systems domain documents"}
        )
        
        logger.info(f"Vector database initialized at {self.vector_db_path}")
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the data directory"""
        documents = []
        
        logger.info(f"Loading documents from {self.data_dir}")
        
        for file_path in tqdm(self.data_dir.glob("*.txt"), desc="Loading documents"):
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
                # Clean up the line
                line = re.sub(r'^\d+\s*', '', line)  # Remove leading numbers
                line = re.sub(r'[^\w\s\-\.]', '', line)  # Keep only alphanumeric, spaces, hyphens, dots
                if line:
                    return line
        
        # Fallback to filename
        return filename.replace('_', ' ').title()
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split document into overlapping chunks"""
        chunks = []
        sentences = sent_tokenize(document.content)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk_doc = Document(
                    id=f"{document.id}_chunk_{chunk_index}",
                    content=current_chunk.strip(),
                    title=document.title,
                    source=document.source,
                    chunk_id=f"{document.id}_chunk_{chunk_index}",
                    chunk_index=chunk_index
                )
                chunks.append(chunk_doc)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_doc = Document(
                id=f"{document.id}_chunk_{chunk_index}",
                content=current_chunk.strip(),
                title=document.title,
                source=document.source,
                chunk_id=f"{document.id}_chunk_{chunk_index}",
                chunk_index=chunk_index
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last part of text for overlap"""
        words = text.split()
        overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
        return " ".join(overlap_words)
    
    def build_vector_index(self, documents: Optional[List[Document]] = None):
        """Build the vector index from documents"""
        if documents is None:
            documents = self.load_documents()
        
        logger.info("Building vector index...")
        
        # Check if collection already has documents
        if self.collection.count() > 0:
            logger.info("Vector index already exists. Clearing and rebuilding...")
            # Delete the existing collection and recreate it
            self.client.delete_collection("llm_systems_docs")
            self.collection = self.client.get_or_create_collection(
                name="llm_systems_docs",
                metadata={"description": "LLM Systems domain documents"}
            )
        
        all_chunks = []
        chunk_metadata = []
        chunk_ids = []
        
        for doc in tqdm(documents, desc="Processing documents"):
            chunks = self.chunk_document(doc)
            
            for chunk in chunks:
                all_chunks.append(chunk.content)
                chunk_metadata.append({
                    "document_id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "chunk_index": chunk.chunk_index
                })
                chunk_ids.append(chunk.chunk_id)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        
        # Add to ChromaDB in batches to avoid batch size limits
        logger.info("Adding to vector database...")
        batch_size = 1000  # Safe batch size for ChromaDB
        
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(all_chunks), batch_size), 
                     desc="Adding to vector DB", 
                     total=total_batches):
            end_idx = min(i + batch_size, len(all_chunks))
            
            batch_chunks = all_chunks[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadata = chunk_metadata[i:end_idx]
            batch_ids = chunk_ids[i:end_idx]
            
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        
        logger.info(f"Vector index built with {len(all_chunks)} chunks")
    
    def search(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of QueryResult objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Convert to QueryResult objects
        query_results = []
        for i in range(len(results['ids'][0])):
            result = QueryResult(
                document_id=results['metadatas'][0][i]['document_id'],
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                score=results['distances'][0][i],
                metadata=results['metadatas'][0][i]
            )
            query_results.append(result)
        
        return query_results
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Get a document by its ID"""
        # Search for chunks of this document
        results = self.collection.get(
            where={"document_id": document_id}
        )
        
        if not results['ids']:
            return None
        
        # Reconstruct document from chunks
        chunks = []
        for i, chunk_id in enumerate(results['ids']):
            chunks.append({
                'index': results['metadatas'][i]['chunk_index'],
                'content': results['documents'][i]
            })
        
        # Sort by chunk index and combine
        chunks.sort(key=lambda x: x['index'])
        full_content = '\n\n'.join([chunk['content'] for chunk in chunks])
        
        return Document(
            id=document_id,
            title=results['metadatas'][0]['title'],
            content=full_content,
            source=results['metadatas'][0]['source']
        )

def main():
    """Main function to demonstrate the RAG system"""
    # Initialize RAG system
    rag = LLMSystemsRAG()
    
    # Build vector index
    rag.build_vector_index()
    
    # Example queries
    test_queries = [
        "What is FlashAttention and how does it work?",
        "How does distributed training work for large language models?",
        "What are the different types of parallelism in LLM training?",
        "How does quantization help with LLM inference?",
        "What is the transformer architecture?"
    ]
    
    print("\n" + "="*80)
    print("RAG SYSTEM DEMONSTRATION")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        results = rag.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document: {result.metadata['title']}")
            print(f"   Score: {result.score:.4f}")
            print(f"   Content: {result.content[:200]}...")
            print(f"   Source: {result.metadata['source']}")

if __name__ == "__main__":
    main()
