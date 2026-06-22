"""
Store embeddings in a FAISS vector index with associated chunk data.
"""

import faiss
import pickle


def store_embeddings(embedding_matrix, chunks):
    """
    Create a FAISS index from embeddings and save the index and chunk metadata.

    Args:
        embedding_matrix (np.ndarray): Array of document embeddings.
        chunks (list): Document chunks with associated metadata.

    Returns:
        None
    """

    # Get embedding vector size
    dimension = embedding_matrix.shape[1]

    # Create FAISS similarity search index
    index = faiss.IndexFlatL2(dimension)

    index.add(embedding_matrix)

    print(f"Total vectors stored: {index.ntotal}")

    # Save vector index for future searches
    faiss.write_index(index, "embeddings.index")

    # Save chunk text and metadata
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Saved FAISS index + chunks ✔")