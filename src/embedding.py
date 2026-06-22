"""
Generate vector embeddings for text chunks using SentenceTransformer.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def create_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks (list): Chunk dictionaries containing a "text" field.

    Returns:
        numpy.ndarray: Embedding matrix of shape (n_chunks, embedding_dim).
    """

    print("Loading local embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = []

    print("Creating embeddings...")

    for i, chunk in enumerate(chunks):

        vector = model.encode(chunk["text"])

        embeddings.append(vector)

        if i % 50 == 0:
            print(f"Processed {i}/{len(chunks)} chunks")

    # Convert embeddings into a FAISS-compatible matrix
    embedding_matrix = np.array(embeddings).astype("float32")

    print("\nEmbedding complete ✔")

    return embedding_matrix