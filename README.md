# RAG System

A Retrieval Augmented Generation (RAG) application with a Streamlit UI and retrieval and ingestion modularised and performance tested.

The ingestion processes involves loading PDF lecture notes into text, chunking them, embedding them as vectors in an indexed FAISS vector database then finally storing the chunks and the indexes in a folder for use in retrieval. 

In retrieval, the ingestion data and vector database are loaded. A user question is embedded as a vector and a similarity search is performed with it against all chunks in the vector database. The top K chunks are returned as context to a local LLM which also receives the question and a master prompt. The LLM returns an answer which is passed and displayed on the UI. 

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
- Streamlit web interface - (main.py)
- Modular ingestion pipeline - (ingestion.py)

## Motivations
Studying physics at Bristol, I used chatbots like ChatGPT and Claude by Anthropic as tools for revision. However, as course content speciallised more context was needed for accurate answers to be returned. RAG meant I could upload all of my lecture slides once for unlimited quick and accurate course-specific responses. 

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
src/
├── ingestion.py
├── 
├── 
└── 
README.md
requirements.txt
.gitignore
```
## Design Choices

### FAISS
Chosen as a lightweight local vector database for fast semantic search and easy integration with Python.

### Ollama
Allows local LLM inference without external API costs or dependency on cloud providers.

### Streamlit
Used to rapidly prototype a user interface.

## Installation

1. Clone the repository

```bash
git clone https://github.com/george-powell/RAG-system
cd RAG-system
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Install Ollama and pull the model you want

```bash
ollama pull llama3.1
```

5. Build the vector database

```bash
python src/ingestion.py
```

## Running the application

Start Ollama:

```bash
ollama serve
```

Then run the UI:

Streamlit:

```bash
streamlit run main.py
```

## Future Improvements

I started this project to develop my understanding of RAG's back-end. Hence, whilst a good front-end is necessary for user experience, I plan to engage more with developments in back-end approaches to RAG. The repository will likely evolve in accordance to this. I am currently learning about other methods such as vectorless RAG, trying to implementing a cheap reasoning-based retreival system, and most interesting of all, proxy-pointer RAG. My other future ambitions for the project are to:
1. Improve flask UI
2. Persistent user accounts and chat history
3. React frontend
4. Past paper generation
5. Develop into a university-tutor SaaS

## Screenshots
I provided PDF documents from my provided university lecture notes, asking the content-relevant question: "Is the Hamiltonian a unitary operator?"
![Alt text](images/streamlit.png)
