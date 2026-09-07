"""
Class with retriever and Ollama model passed in. Class contains the get_response function, 
where a question is compared to the vector database and the top K similar chunks are pasted 
as context for Ollama's llama3.1 model to answer the question. The answer is returned to be 
displayed on a UI.
"""

import ollama
import time


class RAG:
    """
    Contains get_response function, where a question is compared to the vector database 
    and the top K similar chunks are pasted as context for Ollama's llama3.1 model to answer the question.
    The answer is returned to be displayed on a UI.
    """

    def __init__(self, retriever, model="llama3.1"):
        self.retriever = retriever
        self.model = model

    def get_response(self, question, extra_info=False):
        """
        For a given question, passes the K most relevant chunks 
        as context to an LLM to answer the question.

        Parameters
        ----------
        question : str
            Question to be answered by the LLM.
        extra_info : bool
            Wether performance metrics are returned.

        Returns
        -------
        answer : str
            LLM answer to question given context provided.
        """
        start_time = time.perf_counter()
        context, indices = self.retriever.retrieve(
            question,
            return_indices=True
        )
        end_time = time.perf_counter()
        retrieval_latency = end_time - start_time

        prompt = f"""
You are a helpful assistant.

Rules:
- Use clear section headers
- Use bullet points for steps
- Explain intuition clearly
- Keep answers structured and easy to follow
- Use simple readable maths
- Avoid raw LaTeX delimiters
- Only answer using the provided context
- If the question is outside the scope of the context,
  reply: "Question outside scope, do you wish to continue?"

CONTEXT:

{context}

QUESTION:

{question}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        if extra_info:
            return answer, indices, retrieval_latency

        return answer