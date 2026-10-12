"""
Evaluates the retrieval performance of the RAG application using a command line interface.
"""

import time
import pandas as pd
from src.retrieval import Retriever
from src.RAG import RAG


def main():

    recall_questions = {
    "Which officials can ask a Standing Review Board to carry out reviews beyond the required life-cycle reviews?": 106,
    "What do Pre-Phase A conceptual designs aim to establish about a proposed mission?": 125,
    "What does the Phase A team revisit to confirm the project is justified and practical enough to earn a place in NASA's budget?": 127,
    "What is refined in Phase A so the cost impacts of competing design alternatives can be compared?": 132,
    "Over which project phases are the system design processes primarily applied?": 204,
    "Which stakeholders are listed as examples for the Pre-Phase A life-cycle stage?": 214,
    "What does a complete description of a product's functions help potential users determine?": 263,
    "Which section of the NASA Expanded Guidance for Systems Engineering gives a more detailed discussion of technology assessment during design concept development?": 303,
    "When mission success can't be captured in a single combined measure, what should the systems engineer weigh alongside the quantitative data?": 323,
    "What benefit does baselining a particular design solution give the technical team?": 338,
    "What should the developmental test program include to capture data on system interactions?": 381,
    "In which project phases can design and analysis be handled by a single general discipline rather than multiple specialized ones?": 529,
    "What does the basic technical planning strategy say to do with NASA procedural requirements that apply at each level of the PBS?": 532,
    "Why is a qualification unit typically not flown?": 559,
    "At what review is the product baseline typically established?": 658,
    "Why does the handbook recommend a separate data management plan rather than folding it into the program/project plan?": 674,
    "Why do project managers need visibility into progress against the technical plans?": 704,
    "In which phase, and before which key decision point, is the MDR/SDR conducted?": 716,
    "When should the technical team present several closely ranked options to the decision-maker instead of a single recommendation?": 754,
    "What is the term for the measure of how readily an item can be kept in or restored to specified conditions by maintainers of a given skill level?": 818,
    "Where are the success criteria for a technical review documented?": 854,
    "What does system safety engineering aim to achieve?": 858,
    "What question does the handbook suggest asking to uncover the real need behind a requirement?": 877,
    "What examples of methods does the SEMP outline give for supporting technical effort integration?": 990,
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


    for question, relevant_chunk in recall_questions.items():

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
                relevant_chunk in indices[:k]
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