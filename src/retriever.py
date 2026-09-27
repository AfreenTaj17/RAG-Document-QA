from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


def retrieve_documents(query, k=3):
    vector_store = load_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


if __name__ == "__main__":
    query = "What is machine learning?"

    results = retrieve_documents(query)

    print(f"Query: {query}")
    print(f"\nNumber of retrieved documents: {len(results)}")

    for i, document in enumerate(results, start=1):
        print(f"\n--- RESULT {i} ---")
        print(f"Page: {document.metadata.get('page')}")
        print(document.page_content[:500])
