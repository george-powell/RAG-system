"""
Retrieves relevant document content for user questions
using a FAISS vector database and SentenceTransformer embeddings.
"""

import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    """
    Contains retrieve function, retrieving the top k relevant chunks for a given question.
    """

    def __init__(self):
        self.vector_database = faiss.read_index(
            "retrieval_data/embeddings.index"
        )

        with open("retrieval_data/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, question, k=5, return_indices=False):

        """
        Retrieve the most relevant chunks for a user question.

        The question is converted into an embedding using the
        same SentenceTransformer model used during ingestion.
        FAISS then searches the vector database for the k most
        similar chunks.

        Parameters
        ----------
        question : str
            User's question.
        k : int
            Number of relevant chunks to retrieve.

        Returns
        -------
        context : str
            Retrieved content combined into a single context string.
        """
        
        # Convert question into embedding vector
        question_vector = self.embedder.encode([question]).astype("float32")

        # Retrieve closest chunks
        distances, indices = self.vector_database.search(question_vector, k)

        retrieved_chunks = [self.chunks[i]["text"] for i in indices[0]]

        context = "\n\n".join(retrieved_chunks)

        if return_indices:
            return context, indices[0]
    
        return context
