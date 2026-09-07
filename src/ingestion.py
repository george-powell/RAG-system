"""
RAG Ingestion Pipeline.

Loads the PDFs into text, chunks them then embeds them 
as vectors in a FAISS database. The chunks and their
corresponding vector indexes are saved to 'retrieval_data/'
"""

from pypdf import PdfReader
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import time

def load_pdf(folder):
    """
    Load PDF files from a folder and extract page text.

    Parameters
    ----------
    folder : str
        Directory containing PDF files.

    Returns
    -------
    documents : list[dict]
        Document dictionaries containing 'text', 'source' file, and 'page' number.
    """
    documents = []

    # Process each PDF in the folder by alphabetical order
    pdf_counter = 0
    for file in sorted(os.listdir(folder)):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"Loading: {path}")

            reader = PdfReader(path)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()

                if text:
                    pdf_successful = True
                    # Store page text with source metadata
                    documents.append({
                        "text": text,
                        "source": file,
                        "page": page_num + 1
                    })
            # Track number of PDFs processed
            if pdf_successful:
                pdf_counter += 1

    if not documents:
        raise ValueError("No PDF documents found.")
    
    print(f"\nTotal PDFs loaded: {pdf_counter}")
    print(f"Total pages loaded: {len(documents)}")
    return documents

def chunk_documents(documents, CHUNK_SIZE=1000, CHUNK_OVERLAP=200):
    """
    Split documents into overlapping chunks and preserve source metadata.

    Parameters
    ----------
    documents : list[dict]
        Document dictionaries containing 'text', 'source' file, and 'page' number.
    CHUNK_SIZE : int
        Maximum characters per chunk.
    CHUNK_OVERLAP : int
        Number of overlapping characters between chunks.

    Returns
    -------
    chunks : list[dict]
        Chunk dictionaries containing 'text', 'source', and 'page'.
    """
    chunks = []

    for doc in documents:
        text = doc["text"]

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE

            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })

            # Move forward while keeping overlap between chunks
            start += CHUNK_SIZE - CHUNK_OVERLAP

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"Total chunks created: {len(chunks)}")

    return chunks

def create_embeddings(chunks):
    """
    Generate embeddings using SentenceTransformer's 
    all-MiniLM-L6-v2 model for a list of text chunks.

    Parameters
    ----------
    chunks : list[dict]
        Chunk dictionaries containing 'text', 'source', and 'page'.

    Returns
    -------
    embedding_matrix : numpy.ndarray
        Embedding matrix of shape (n_chunks, embedding_dim).
    """

    print("Loading local embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    # Convert embeddings into a FAISS-compatible matrix
    embedding_matrix = embeddings.astype("float32")

    n_chunks, embedding_dim = embedding_matrix.shape

    print(
        f"\nCreated {embedding_dim} dimensional embedding vectors "
        f"for {n_chunks} chunks."
    )

    return embedding_matrix

def store_embeddings(embedding_matrix, chunks, retrieval_data_path):
    """
    Create a FAISS index from embeddings and save the index and chunk metadata.

    Parameters
    ----------
    embedding_matrix : numpy.ndarray
        Embedding matrix of shape (n_chunks, embedding_dim).
    chunks : list[dict]
        Chunk dictionaries containing 'text', 'source', and 'page'.
    
    Note: 
        To be compatable with FAISS, embedding_matrix must be passed as type float32.
    """

    # Create FAISS similarity search index of embeddings
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])

    index.add(embedding_matrix)

    print(f"Total vectors stored in index: {index.ntotal}")

    index_path = os.path.join(retrieval_data_path, "embeddings.index")
    chunks_path = os.path.join(retrieval_data_path, "chunks.pkl")

    # Save vector index for future searches
    faiss.write_index(
        index, 
        index_path)

    # Save chunk text and metadata
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("Saved FAISS index + chunks")

    retrieval_data_size = os.path.getsize(index_path) + os.path.getsize(chunks_path)

    print(f"Retrieval data size: {retrieval_data_size / (1024**2):.2f} MB")

def main():
    """
    Run the document ingestion pipeline.

    Loads lecture PDFs, chunks the text, generates embeddings,
    and stores them in the FAISS vector database.
    """

    start_time = time.time()

    pwd = os.getcwd()
    retrieval_data_path = os.path.join(pwd, "retrieval_data")
    os.makedirs(retrieval_data_path, exist_ok=True)

    # Load source lecture documents
    documents = load_pdf("data")

    # Split documents into chunks for retrieval
    chunks = chunk_documents(documents)

    # Generate vector embeddings for each chunk
    embedding_matrix = create_embeddings(chunks)

    # Store embeddings and metadata for semantic search
    store_embeddings(embedding_matrix, chunks, retrieval_data_path)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Ingestion time: {total_time:.2f}s")


if __name__ == "__main__":
    main()