import streamlit as st

from src.rag_pipeline import rag_pipeline


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


st.title("📚 RAG-Based Document Q&A")

st.write(
    "Ask questions about the machine learning study material "
    "using Retrieval-Augmented Generation."
)


st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of documents to retrieve",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.write("**RAG Pipeline**")
st.sidebar.write("📄 Document")
st.sidebar.write("↓")
st.sidebar.write("🔎 FAISS Retrieval")
st.sidebar.write("↓")
st.sidebar.write("💬 Retrieved Context")


question = st.text_input(
    "Ask a question about the document:",
    placeholder="Example: What is supervised learning?"
)


if question:

    try:

        with st.spinner("Searching the document..."):

            answer, prompt, documents = rag_pipeline(
                question,
                k=top_k,
                use_llm=False
            )


        st.subheader("📖 Retrieved Context")


        if documents:

            for i, document in enumerate(
                documents,
                start=1
            ):

                page = document.metadata.get("page")

                if page is not None:
                    title = f"Source {i} — Page {page + 1}"
                else:
                    title = f"Source {i}"

                with st.expander(title):

                    st.write(
                        document.page_content
                    )

        else:

            st.info(
                "No relevant document sections were retrieved."
            )


        st.info(
            "🤖 LLM generation is disabled for this demo. "
            "The application uses FAISS semantic retrieval "
            "to find the most relevant document sections."
        )


    except Exception as e:

        st.error(
            "Something went wrong while processing your question."
        )

        st.exception(e)
