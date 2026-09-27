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


def evaluate_retrieval(k=3):

    total = len(test_cases)
    successful = 0

    for test_case in test_cases:

        question = test_case["question"]
        relevant_pages = test_case["relevant_pages"]

        results = retrieve_documents(question, k=k)

        retrieved_pages = [
            document.metadata.get("page")
            for document in results
        ]

        found = any(
            page in relevant_pages
            for page in retrieved_pages
        )

        if found:
            successful += 1

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Expected pages: {relevant_pages}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Relevant document retrieved: {found}")

    recall_at_k = successful / total

    print("\n" + "=" * 70)
    print(f"Recall@{k}: {recall_at_k:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_retrieval(k=3)