from src.retriever import retrieve_documents


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


def rag_pipeline(question, k=3, use_llm=False):

    # Step 1: Retrieve relevant documents
    documents = retrieve_documents(
        question,
        k=k
    )

    # Step 2: Build prompt
    prompt = build_prompt(
        question,
        documents
    )

    # Step 3: LLM is optional
    if use_llm:
        from src.llm import create_llm

        llm = create_llm()
        response = llm.invoke(prompt)
        answer = response.content
    else:
        answer = None

    return answer, prompt, documents
