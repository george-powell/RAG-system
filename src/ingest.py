"""
Run the document ingestion pipeline.

Loads lecture PDFs, chunks the text, generates embeddings,
and stores them in the vector database.
"""

from load_pdf import load_pdf
from chunking import chunk_documents
from embedding import create_embeddings
import vector_store


def main():
    """
    Run the document ingestion pipeline.
    """

    # Load source lecture documents
    documents = load_pdf("data/*.pdf")

    # Split documents into chunks for retrieval
    chunks = chunk_documents(documents)

    # Generate vector embeddings for each chunk
    embedding_matrix = create_embeddings(chunks)

    # Store embeddings and metadata for semantic search
    vector_store.store_embeddings(embedding_matrix, chunks)


if __name__ == "__main__":
    main()