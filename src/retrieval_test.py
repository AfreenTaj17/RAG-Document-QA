from retriever import retrieve_documents


questions = [
    "What is machine learning?",
    "What are the types of machine learning systems?",
    "What is model-based learning?",
    "What is supervised learning?",
    "What is unsupervised learning?"
]


if __name__ == "__main__":

    for question in questions:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        results = retrieve_documents(question, k=3)

        for i, document in enumerate(results, start=1):

            print(f"\n--- RESULT {i} ---")
            print(f"Page: {document.metadata.get('page')}")
            print(document.page_content[:700])