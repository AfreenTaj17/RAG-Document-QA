import mlflow

from retriever import retrieve_documents


test_cases = [
    {
        "question": "What is machine learning?",
        "relevant_pages": [0]
    },
    {
        "question": "What are the types of machine learning systems?",
        "relevant_pages": [2]
    },
    {
        "question": "What is model-based learning?",
        "relevant_pages": [11]
    },
    {
        "question": "What is supervised learning?",
        "relevant_pages": [3]
    },
    {
        "question": "What is unsupervised learning?",
        "relevant_pages": [3, 4, 5]
    }
]


def evaluate_retrieval(k):

    successful = 0

    for test_case in test_cases:

        results = retrieve_documents(
            test_case["question"],
            k=k
        )

        retrieved_pages = [
            document.metadata.get("page")
            for document in results
        ]

        found = any(
            page in test_case["relevant_pages"]
            for page in retrieved_pages
        )

        if found:
            successful += 1

    return successful / len(test_cases)


if __name__ == "__main__":

    mlflow.set_experiment("RAG Retrieval Experiments")

    with mlflow.start_run():

        chunk_size = 1000
        chunk_overlap = 200
        top_k = 3
        embedding_model = "all-MiniLM-L6-v2"

        recall = evaluate_retrieval(top_k)

        mlflow.log_param("embedding_model", embedding_model)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("top_k", top_k)

        mlflow.log_metric("recall_at_3", recall)

        print("MLflow experiment completed.")
        print(f"Recall@3: {recall:.2%}")