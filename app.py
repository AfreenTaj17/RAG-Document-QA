import streamlit as st

from src.retriever import retrieve_documents


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


st.title("📚 RAG-Based Document Q&A")
st.write(
    "Ask questions about the machine learning study material "
    "using semantic search."
)


st.sidebar.header("Settings")

top_k = st.sidebar.slider(
    "Number of documents to retrieve",
    min_value=1,
    max_value=5,
    value=3
)


question = st.text_input(
    "Ask a question about the document:"
)


if question:

    with st.spinner("Searching the document..."):

        documents = retrieve_documents(
            question,
            k=top_k
        )

    st.subheader("Retrieved Context")

    for i, document in enumerate(documents, start=1):

        page = document.metadata.get("page")

        with st.expander(
            f"Context {i} — Page {page + 1}"
        ):
            st.write(document.page_content)

    st.info(
        "LLM generation is currently disabled because "
        "OpenAI API quota is unavailable."
    )