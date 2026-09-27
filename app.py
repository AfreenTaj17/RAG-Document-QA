import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.retriever import retrieve_documents


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


# =====================================================
# HEADER
# =====================================================

st.title("📚 RAG-Based Document Q&A")

st.write(
    "Ask questions about the machine learning study material "
    "and get answers using Retrieval-Augmented Generation."
)


# =====================================================
# OPENAI API KEY
# =====================================================

if "OPENAI_API_KEY" not in st.secrets:
    st.error(
        "OpenAI API key is not configured. "
        "Please add OPENAI_API_KEY to Streamlit Secrets."
    )
    st.stop()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# =====================================================
# LOAD LLM
# =====================================================

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY
    )


llm = load_llm()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of documents to retrieve",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**Pipeline:**"
)

st.sidebar.write(
    "📄 Documents → 🔎 FAISS → 🤖 OpenAI → 💬 Answer"
)


# =====================================================
# USER QUESTION
# =====================================================

question = st.text_input(
    "Ask a question about the document:",
    placeholder="Example: What is machine learning?"
)


# =====================================================
# RAG PIPELINE
# =====================================================

if question:

    with st.spinner("Retrieving relevant information..."):

        try:
            documents = retrieve_documents(
                question,
                k=top_k
            )

        except Exception as e:
            st.error(
                f"Unable to retrieve documents: {str(e)}"
            )
            st.stop()


    if not documents:

        st.warning(
            "No relevant information was found in the document."
        )
        st.stop()


    # =================================================
    # BUILD CONTEXT
    # =================================================

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # =================================================
    # PROMPT
    # =================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available in
the provided document.

Do not invent or assume information.

Keep the answer clear, concise, and easy to understand.

Context:
{context}
"""
            ),
            (
                "human",
                "{question}"
            )
        ]
    )


    # =================================================
    # GENERATE ANSWER
    # =================================================

    with st.spinner("Generating answer..."):

        try:

            chain = prompt | llm

            response = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            answer = response.content

        except Exception as e:

            st.error(
                f"Unable to generate answer: {str(e)}"
            )

            st.info(
                "Please check that your OpenAI API key "
                "has available API quota."
            )

            st.stop()


    # =================================================
    # DISPLAY ANSWER
    # =================================================

    st.subheader("💬 Answer")

    st.success(answer)


    # =================================================
    # DISPLAY SOURCES
    # =================================================

    st.subheader("📖 Retrieved Sources")

    for i, document in enumerate(
        documents,
        start=1
    ):

        page = document.metadata.get("page")

        if page is not None:
            page_number = page + 1
            title = f"Source {i} — Page {page_number}"
        else:
            title = f"Source {i}"

        with st.expander(title):

            st.write(
                document.page_content
            )
