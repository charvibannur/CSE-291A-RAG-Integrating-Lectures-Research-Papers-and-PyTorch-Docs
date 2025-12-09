# RAG-Integrating-Lectures-Research-Papers-and-PyTorch-Docs

This is the project repository for the course Systems for LLM and Agents (CSE291). his study addresses several RAG-specific challenges: 1. documents vary widely in structure (e.g., code snippets, formulas, pdfs, and slide text), terminology overlapping exists across engineering contexts and the information needed to answer graduate-level questions often spans across multiple documents.

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/6167a3c9-cd1b-4178-8730-4a841b488838" />

## Repository Structure
``` bash
├── assets/ # Figures, images, and supporting visuals
├── baselines/ # BM25 lexical retriever and a semantic vector-based RAG baseline
├── data/ # Input documents (cleaned .txt files)
├── evaluator/ # Evaluation scripts for retrieval metrics
├── queries/ # Manually curated queries and gold reference answers
├── results/ # Retrieved passages and evaluation outputs
├── .gitignore
└── README.md
```

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/charvibannur/CSE-291A-RAG-Integrating-Lectures-Research-Papers-and-PyTorch-Docs.git
cd CSE-291A-RAG-Integrating-Lectures-Research-Papers-and-PyTorch-Docs.git
```

### Set Up Environment
We recommend using Python 3.9+ and creating a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
The docs for the RAG system are present in the \data. The queries for single doc and multi doc retrieval are present in the \queries folder.

### Baseline 1: BM25
BM25 is a search algorithm that evaluates the similarity of a document with the query. The BM25 algorithm lexically tries to find out how often the word appears in document (term frequency) and across all the pdfs. The top 3 most relevant document name and match text is returned. 
```bash
jupyter nbconvert --to notebook --execute bm25_baseline.ipynb --output rag_baseline1_output.ipynb
```
Results are automatically saved as a json.

### Baseline 2: Semantic Search
Semantic search is built by dividing each document into small overlapping chunks. Embeddings are then generated using the all-MiniLM-L6-v2 Sentence Transformer model. These embeddings are stored in the vector database along with the metadata; we use ChromaDB for this study. For each query, the system retrieves top-k most relevant chunks using cosine similarity.
```bash
python rag_semanticsearch.py
```
Results are automatically stored as a json.

### Proposed Methodology: Hybrid RAG
A RAG that uses a recursive and separator-driven chunking mechanism that adapts to the structure of technical documents. This produces contextually rich chunks that are suitable for retrieving technical documents, dense + sparse retrieval strategy with a unified candidate pool to leverage the complementary strengths of using dense embeddings and sparse BM25 vectors and a neural cross-encoder to rerank based on joint query and document pairs representing a cascading architecture.

Export the OpenAI key
```
export OPENAI_API_KEY="your_api_key_here"
```
or pass it as a flag
```
--openai-key "your_api_key_here"
```

Modify the data directory in the /model/hybrid_rag.py script and make sure all the documents are .txt files.
```
rag = LLMSystemsRAG(data_dir="my_documents")
```
The hybrid_rag.py file has an evaluator as well. Make sure to add your query files to the query folder respectively.
```
# Run Evaluation
queries_dir = Path("queries")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
    
# Combine queries for evaluation
multi_file = queries_dir / "multi_file_retrieval_queries.csv" # Change path
single_file = queries_dir / "single_file_retrieval_queries.csv" # Change path

```
An example of the queries.csv files are present in the queries folder.

To run the RAG pipeline use the following command.
```
python model/script.py \
    --openai-key $OPENAI_API_KEY \
    --rag-script model/hybrid_rag.py \
    --eval-script evaluator/eval.py
```

### Evaluator:
Our custom evaluator suits consist of a list of metrics that prioritise retrievals that are factually correct and contextually aligned with the gold standard.
```bash
jupyter nbconvert --to notebook --execute evaluator.ipynb
```

Result files are report.txt, summary.json and a per query .csv file.

## Authors
1. Charvi Bannur
2. Shivani Chinta
3. Sathvik Bhaskarpandit
4. Aryan Dokania
5. Loay Rashid
