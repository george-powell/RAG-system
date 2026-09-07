"""
Evaluates the retrieval performance of the RAG application using a command line interface.
"""

import time
import pandas as pd
from src.retrieval import Retriever
from src.RAG import RAG


def main():

    recall_questions = {
    "How is a quantum state represented mathematically?": [312],
    "What is meant by a superposition of quantum states?": [314, 315],
    "What condition must a quantum state satisfy in order to be normalised?": [314],
    "What is the physical interpretation of the coefficients in a quantum state written in a particular basis?": [317],
    "What is the inner product of two quantum states, and what is it used for?": [327, 328],
    "How is the bra corresponding to a ket constructed when the ket contains complex coefficients?": [331],
    "What exercise asks you to calculate the inner product between |ψ⟩ = (3/5)|L⟩ + (4/5)|R⟩ and |ϕ⟩ = (4/5)|L⟩ - (3/5)|R⟩.": [332],
    "How is the Kronecker delta function defined mathematically?": [334],
    "What does the operator |R⟩⟨L| do to the state |L⟩?": [337],
    "What is the identity operator for a two-dimensional system in the |L⟩, |R⟩ basis?": [338],
    "How can operators be represented as matricies?": [339],
    "What condition must an operator satisfy to be Hermitian?": [344],
    "Determine which of the following operators are Hermitian: A = 2|L⟩⟨L| + 3i|R⟩⟨R|, B = i|L⟩⟨R| - i|R⟩⟨L|, C = 2|L⟩⟨L| + 3|R⟩⟨R|.": [344, 345],
    "What are the possible outcomes when measuring an observable represented by an operator?": [350, 351],
    "What is the general summation equation for an operator with an orthonormal eigenbasis?": [350, 351],
    "What is the time-evolution operator for a quantum system whose Hamiltonian has energy eigenstates |n⟩ with energies E_n?": [352, 353],
    "Why is time evolution in quantum theory unitary?": [359, 360],
    "What exercise asks you to show that unitary time evolution preserves the inner product between two quantum states.": [361],
    "For a quantum state expressed as a superposition of energy eigenstates, how does each coefficient change with time?": [361, 362],
    "How does a measurement affect the quantum state after an outcome has been obtained?": [371, 372],
    "When was the Stern-Gerlach experiment performed?": [400],
    "What are the possible results of measuring the z-component of spin for a spin-1/2 particle?": [402, 414],

    "How is the instantaneous velocity of an object mathematically defined?": [521, 522, 523],
    "Paste in the contents of section 3.2.1 Basis unit vectors.": [539, 540],
    "What is an inertial reference frame?": [547],
    "How is the centre of mass frame defined?": [553, 554],
    "Derive the rocket equation in its derivative form": [563, 564, 565, 566],
    "What is the scalar product also known as?": [570],
    "What are statics problems in mechanics?": [577],
    "Is the coefficient of static friction or kinetic friction typically larger?": [590],
    "How is the two-body problem effectively reduced to a one-body problem?": [629, 630, 631, 632],
    "How is a rigid body defined?": [642],
    "How is circular motion produced?": [649],
    "Derive the analogue to Newton's second law for circular motion.": [657, 658, 659],

    "What is the difference between frequency and angular frequency?": [35],
    "What two assumptions are made to solve the damped SHM differential equation?": [57, 58],
    "How is critical damping mathematically defined?": [61],
    "What name is given to the measure of the persistance of an oscillation?": [64],
    "What two regimes does any forced oscillation consist of?": [66, 67],
    "How does Q-factor relate to full-width at half-maximum of an average power against driving drequency plot for forced oscillations?": [70, 71]
}

    retrieval_times = []
    response_times = []
    recall_results = {
        "Recall@5": [],
        "Recall@3": [],
        "Recall@1": []
    }

    retriever = Retriever()

    rag = RAG(retriever)


    for question, relevant_chunks in recall_questions.items():

        print(f"Evaluating: {question}")

        # Measure end-to-end response latency
        start_time = time.perf_counter()

        answer, indices, retrieval_time = rag.get_response(question, extra_info = True)

        end_time = time.perf_counter()
        response_time = end_time - start_time

        retrieval_times.append(retrieval_time)
        response_times.append(response_time)

        # calculate recall@K
        for k in [1, 3, 5]:
            recall_results[f"Recall@{k}"].append(
                len(set(relevant_chunks) & set(indices[:k])) / len(relevant_chunks)
                )

    # Calculate evaluation statistics
    n_questions = len(recall_questions)

    mean_retrieval_time = sum(retrieval_times) / n_questions
    mean_response_time = sum(response_times) / n_questions

    # Store results in a DataFrame
    results = pd.DataFrame({
        "Metric": [
            "Number of questions",
            "Mean retrieval latency (s)",
            "Mean end-to-end latency (s)",
            "Recall@1",
            "Recall@3",
            "Recall@5"
        ],
        "Result": [
            round(n_questions),
            mean_retrieval_time,
            mean_response_time,
            sum(recall_results["Recall@1"]) / n_questions,
            sum(recall_results["Recall@3"]) / n_questions,
            sum(recall_results["Recall@5"]) / n_questions
        ]
    })

    print("\nEvaluation Results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()