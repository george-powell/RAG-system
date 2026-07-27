"""
Flask web application for the AI Physics Tutor.

Loads the FAISS index and lecture note chunks, retrieves relevant
content for user questions, and generates responses using Ollama.
"""

import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama
from flask import Flask, request, render_template

# Load vector index used for semantic search
index = faiss.read_index("embeddings.index")

# Load chunked lecture note data
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Free embedding model used to encode user questions
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Temporary in-memory chat history
chat_history = []

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handle user questions and render the tutor interface.

    Retrieves relevant lecture content, builds a RAG prompt,
    sends it to Ollama, and displays the response.
    """
    if request.method == 'POST':

        question = request.form.get("question")

        if not question:
            return render_template(
                "index.html",
                chat_history=chat_history,
                error="Please enter a question."
            )

        # Convert question into an embedding vector
        q_vector = embedder.encode([question]).astype("float32")

        # Retrieve the most relevant lecture chunks
        similarity, indices = index.search(q_vector, k=5)

        # Reject questions with insufficient lecture-note relevance
        if similarity[0][0] > 0.3: # Greater than (>) due to FAISS taking argmin of the Euclidian distance (L2)
            return render_template(
            "index.html",
            chat_history=chat_history,
            error="Not enough relevant lecture content found."
            )

        retrieved_chunks = [chunks[i]["text"] for i in indices[0]]

        # Combine retrieved chunks into the RAG context
        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
        You are a helpful physics tutor.

        Rules:
        - Use clear section headers
        - Use bullet points for steps
        - Write equations in clean readable math form
        - Avoid raw LaTeX delimiters
        - Always explain intuition

        LECTURE NOTES:
        {context}

        QUESTION:
        {question}
        """

        # Generate a response using the local LLM
        response = ollama.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": prompt}]
        )

        # Store the question and answer in the chat history
        answer = response["message"]["content"]
        chat_history.append({"question": question, "answer": answer})
    
    return render_template(
        "index.html",
        chat_history=chat_history
    )


@app.route('/clear', methods=['POST'])
def clear_chat():
    """
    Clear the chat history and reload the chat interface.
    """
    chat_history.clear()
    return render_template(
        "index.html",
        chat_history=chat_history
    )

# FUTURE CHANGE: Set debug=False in production.
# Debug mode automatically reloads the server when code changes.
if __name__ == '__main__':
    app.run(debug=True)
