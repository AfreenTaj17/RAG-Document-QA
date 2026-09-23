from rag_pipeline import rag_pipeline


questions = [
    "What is machine learning?",
    "What are the types of machine learning systems?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is model-based learning?"
]


if __name__ == "__main__":

    for question in questions:

        answer, prompt, documents = rag_pipeline(
            question,
            use_llm=False
        )

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        print("\nRetrieved context:")

        for i, document in enumerate(documents, start=1):
            print(f"\n--- Context {i} | Page {document.metadata.get('page')} ---")
            print(document.page_content[:500])

        print("\n--- LLM STATUS ---")
        print("Skipped because OpenAI API quota is currently unavailable.")