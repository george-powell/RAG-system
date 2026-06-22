"""
Command-line interface for the AI Physics Tutor.

Loads the FAISS vector database, retrieves relevant lecture content
for user questions, and generates responses using Ollama.
"""

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama


def main():
    """
    Run the interactive physics tutor chat loop.
    """

    # Load semantic search index
    index = faiss.read_index("embeddings.index")

    # Load chunk text and metadata
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load embedding model for user questions
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("AI Tutor ready. Ask a physics question!\n")

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        # Convert question into embedding vector
        q_vector = embedder.encode([question]).astype("float32")

        # Retrieve closest lecture chunks
        D, I = index.search(q_vector, k=5)

        retrieved_chunks = [chunks[i]["text"] for i in I[0]]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful physics tutor.

Rules:
- Use clear section headers
- Use bullet points for steps
- Write equations in clean readable math form
- Avoid raw LaTeX delimiters
- Prefer plain-text math unless necessary
- Always explain intuition after derivation

LECTURE NOTES:
{context}

QUESTION:
{question}
"""

        # Generate answer using local LLM
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}]
        )

        print("\nTutor:", response["message"]["content"], "\n")


if __name__ == "__main__":
    main()