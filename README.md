# Local RAG Application

A modular Retrieval-Augmented Generation (RAG) application for querying PDF content using semantic search and a locally hosted large language model (LLM). Built with Python, FAISS, Sentence Transformers, Ollama and Streamlit.

The project implements a complete document [`ingestion`](https://github.com/george-powell/RAG-system/blob/main/src/ingestion.py) and [`retrieval`](https://github.com/george-powell/RAG-system/blob/main/src/retrieval.py) pipeline, with a dedicated [`evaluation script`](https://github.com/george-powell/RAG-system/blob/main/retrieval_evaluation.py) to measure retrieval quality and response latency.

The ingestion pipeline extracts text from PDF documents, splits it into chunks, generates vector embeddings and builds a FAISS index for similarity search. The chunks and index are then saved to disk for use during retrieval. 

In retrieval, the ingestion data and vector database are loaded. The application embeds the user's question and searches the FAISS index for the top-k most similar chunks. These chunks are combined with the question and a system prompt to form the context supplied to a locally hosted LLM. The generated response is then displayed in the Streamlit interface.

## Performance

| Metric | Result |
|---|---:|
| Pages processed | 297 |
| Chunks generated | 1189 |
| Ingestion time | 24.08 s |
| Retrieval data size | 2.73 MB |
| Mean retrieval latency | 64.2 ms |
| Mean end-to-end latency | 5.91 s |
| Recall@1 | 45.8% (11/24) |
| Recall@3 | 62.5% (15/24) |
| Recall@5 | 70.8% (17/24) |

Retrieval was evaluated using 24 manually constructed questions with manually labelled relevant chunks.
Recall@k measures the proportion of questions for which a relevant chunk appears among the top k retrieved results.
Retrieval latency includes query embedding and FAISS search, while end-to-end latency includes retrieval and local LLM generation.
These results provide a useful baseline as reranking and other techniques are later explored.

## What is Retrieval Augmented Generation (RAG) and why is it useful?

Large Language Models (LLMs) often need relevant context provided from large databases or document collections. However, supplying an entire document collection or database for every user query is computationally expensive and often exceeds practical context limits.

Retrieval Augmented Generation (RAG) addresses this problem by first processing the source documents offline. Documents are loaded, split into smaller chunks, and converted into vector embeddings that capture the semantic meaning of the text. These embeddings are then stored in a vector database.

When a user asks a question, the question is embedded using the same embedding model and compared against the stored document embeddings. The most relevant chunks are retrieved and provided to the LLM as context. The LLM can then generate an answer using only the information most relevant to the user's query.

This approach significantly reduces the amount of text supplied to the model while improving the relevance and accuracy of responses. In this project, pdf documents are embedded using the `all-MiniLM-L6-v2` model, stored in a FAISS vector index, and supplied to a local Ollama-hosted LLM. 

## Features
- Local RAG pipeline using FAISS for vector storage
- Semantic search over lecture notes
- Local embeddings using `all-MiniLM-L6-v2`
- Local LLM inference using Ollama's `llama 3.1` model
- Streamlit web interface - [`main.py`](https://github.com/george-powell/RAG-system/blob/main/main.py)
- Modular ingestion pipeline - [`ingestion.py`](https://github.com/george-powell/RAG-system/blob/main/src/ingestion.py)
- Retrieval performance evaluation - [`retrieval_evaluation.py`](https://github.com/george-powell/RAG-system/blob/main/retrieval_evaluation.py)

## Motivation
While studying Physics at the University of Bristol, I used LLMs to support revision. As course material became more specialised, generating reliable answers required access to the specific definitions, explanations and conventions used in lecture notes.

This motivated me to build a local RAG application that retrieves relevant course material before generating an answer. Rather than repeatedly supplying entire documents, the application retrieves a small set of relevant passages for each query.

The project also gave me experience evaluating a software system experimentally, using retrieval-quality metrics and latency measurements to assess its performance.

## Pipeline

| Ingestion Pipeline | Query Pipeline |
|-------------------|----------------|
| PDF Lecture Notes | User Question |
| ↓ | ↓ |
| Document Loading | Embedding |
| ↓ | ↓ |
| Chunking | Similarity Search |
| ↓ | ↓ |
| Embeddings | Top K Relevant Chunks |
| ↓ | ↓ |
| FAISS Index | Prompt + Context |
| | ↓ |
| | LLM Response |

## Project Structure
```
main.py
retrieval_evaluation.py
data/  # contains PDFs
retrieval_data/ 
src/
├── ingestion.py
├── retrieval.py
└── RAG.py
README.md
requirements.txt
.gitignore
```
## Design Choices

- **FAISS:** Provides efficient vector similarity search through a lightweight local index, avoiding the need for a separately hosted vector database service.

- **Sentence Transformers:** The all-MiniLM-L6-v2 model produces compact embeddings suitable for semantic similarity search. Document chunks and user queries are embedded using the same model to ensure they occupy the same vector space.

- **Ollama:** Runs the LLM locally, avoiding per-request cloud inference costs and allowing document context to remain within the local inference workflow.

- **Streamlit:** Provides a lightweight Python-based interface without requiring a separate frontend framework.

## Installation

The application requires the dependencies listed in **requirements.txt**, a local Ollama installation with a compatible language model, and source PDFs from which to build the retrieval index.

1. **clone the repository**
```
git clone https://github.com/george-powell/RAG-system
cd RAG-system
```
2. **Install Ollama and pull the model**
```
ollama pull llama3.1
```

3. **Build the vector database**
place source PDFs in data/ then run:
```
python src/ingestion.py
```

4. **Run the application**

```
ollama serve
streamlit run main.py
```

## Future Improvements

- **Improve retrieval quality:** Evaluate alternative embedding models, chunk sizes and overlap settings.

- **Reranking:** Test a cross-encoder to rerank the initial retrieved candidates and measure its effect on Recall@k and latency.

- **Expand the benchmark:** Increase the number and diversity of questions, verify target chunks and account for multiple valid chunks where appropriate.

- **Reduce response latency:** Investigate prompt length, the number of retrieved chunks, model size and generation settings.

- **Improve answer grounding:** Evaluate whether generated answers are supported by the retrieved passages and handle questions for which the documents provide insufficient evidence.

## Example
After ingesting PDF documents on quantum physics, this is an example of the RAG UI answering a content-relevant question:

![](proof_of_concept.png)
