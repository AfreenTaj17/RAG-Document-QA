from src.retriever import retrieve_documents
from src.llm import create_llm


def build_prompt(question, documents):
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful question-answering assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Do not invent or assume information.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def rag_pipeline(question, k=3, use_llm=True):

    # Step 1: Retrieve relevant documents
    documents = retrieve_documents(
        question,
        k=k
    )

    # Step 2: Build prompt using retrieved context
    prompt = build_prompt(
        question,
        documents
    )

    # Step 3: Generate answer using LLM
    if use_llm:

        llm = create_llm()

        response = llm.invoke(prompt)

        answer = response.content

    else:

        answer = None

    return answer, prompt, documents


if __name__ == "__main__":

    question = "What is machine learning?"

    answer, prompt, documents = rag_pipeline(
        question,
        k=3,
        use_llm=True
    )

    print("\n--- RETRIEVED DOCUMENTS ---\n")

    for i, document in enumerate(
        documents,
        start=1
    ):

        print(f"--- Document {i} ---")
        print(
            f"Page: {document.metadata.get('page')}"
        )
        print(
            document.page_content[:500]
        )
        print()

    print("\n--- PROMPT SENT TO LLM ---\n")
    print(prompt)

    print("\n--- ANSWER ---\n")

    if answer:
        print(answer)
    else:
        print(
            "LLM call skipped."
        )
