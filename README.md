# University RAG Assistant

A Retrieval Augmented Generation (RAG) application that allows university students to ask questions about course-specific lecture notes using a local LLM and semantic search.

The project processes PDF lecture notes, stores them in a vector database, and retrieves relevant content to answer student questions accurately and efficiently.

## Retrieval Augmented Generation - RAG

Large Language Models (LLMs) often need relevant context provided from large databases or document collections. However, supplying an entire document collection or database for every user query is computationally expensive and often exceeds practical context limits.

Retrieval Augmented Generation (RAG) addresses this problem by first processing the source documents offline. Documents are loaded, split into smaller chunks, and converted into vector embeddings that capture the semantic meaning of the text. These embeddings are then stored in a vector database.

When a user asks a question, the question is embedded using the same embedding model and compared against the stored document embeddings. The most relevant chunks are retrieved and provided to the LLM as context. The LLM can then generate an answer using only the information most relevant to the user's query.

This approach significantly reduces the amount of text supplied to the model while improving the relevance and accuracy of responses. In this project, pdf documents are embedded using the `all-MiniLM-L6-v2` model, stored in a FAISS vector index, and supplied to a local Ollama-hosted LLM. 

## Features
- Local RAG pipeline using FAISS for vector storage
- Semantic search over lecture notes
- Local embeddings using `all-MiniLM-L6-v2`
- Local LLM inference using Ollama's `llama 3.1` model
- Command-line interface - (query.py)
- Streamlit web interface - (st_app.py)
- Flask web application prototype - (flask_app.py)
- Modular ingestion pipeline - (ingest.py)

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
| Embeddings | Relevant Chunks |
| ↓ | ↓ |
| FAISS Index | Prompt + Context |
| | ↓ |
| | LLM Response |
## Project Structure

```
src/
├── load_pdf.py
├── chunking.py
├── embedding.py
├── vector_store.py
├── ingest.py
├── query.py
├── st_app.py
└── app.py

data/
└── lecture_notes.pdf

embeddings.index
chunks.pkl
```
## Design Choices

### FAISS
Chosen as a lightweight local vector database for fast semantic search and easy integration with Python.

### Ollama
Allows local LLM inference without external API costs or dependency on cloud providers.

### Streamlit
Used to rapidly prototype a user interface.

### Flask
Used to learn web application fundamentals and provide greater flexibility than Streamlit.

### React
Planned future frontend framework for a production-ready interface.

## Installation

1. Clone the repository

```bash
git clone https://github.com/george-powell/university-RAG-assistant
cd university-RAG-assistant
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

4. Install Ollama and pull the model

```bash
ollama pull llama3.1
```

5. Build the vector database

```bash
python src/ingest.py
```

## Running the application

Start Ollama:

```bash
ollama serve
```

Then choose one interface:

Command line:

```bash
python src/query.py
```

Streamlit:

```bash
streamlit run src/st_app.py
```

Flask:

```bash
python src/app.py
```

## Adaptations of the Interface
query.py:   
Command line chatbot
(useful POC while the backend was being developed)

st_app.py:  
Streamlit interface
(learned interactive UI + state)

flask_app.py:     
Flask web application
(learned web backend architecture and HTML code)

## Future Improvements

I started this project to develop my understanding of RAG's back-end. Hence, whilst a good front-end is necessary for user experience, I plan to engage more with developments in back-end approaches to RAG. The repository will likely evolve in accordance to this. I am currently learning about other methods such as vectorless RAG, trying to implementing a cheap reasoning-based retreival system, and most interesting of all, proxy-pointer RAG. My other future ambitions for the project are to:
1. Improve flask UI
2. Persistent user accounts and chat history
3. React frontend
4. Past paper generation
5. Develop into a university-tutor SaaS

## Screenshots
I provided PDF documents from my provided university lecture notes, asking the content relevant question: "Is the Hamiltonian a unitary operator?"
### command-line Interface
![Alt text](images/command_line.png)

### Streamlit Interface
![Alt text](images/streamlit.png)

### Flask Interface
![Alt text](images/flask.png)
