from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from chunker import chunk_documents
from document_loader import load_pdf


def create_vector_store(chunks, embeddings):
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


if __name__ == "__main__":
    pdf_path = "data/data.pdf"

    # Load PDF
    documents = load_pdf(pdf_path)

    # Create chunks
    chunks = chunk_documents(documents)

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vector_store = create_vector_store(chunks, embeddings)

    # Save vector store
    vector_store.save_local("vectorstore")

    print(f"Total documents: {len(documents)}")
    print(f"Total chunks indexed: {len(chunks)}")
    print("FAISS vector store created successfully.")