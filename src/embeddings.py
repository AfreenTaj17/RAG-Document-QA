from langchain_huggingface import HuggingFaceEmbeddings
from chunker import chunk_documents
from document_loader import load_pdf


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":
    pdf_path = "data/data.pdf"

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    embeddings = create_embeddings()

    print(f"Total chunks: {len(chunks)}")

    # Create an embedding for the first chunk
    vector = embeddings.embed_query(chunks[0].page_content)

    print(f"Embedding dimension: {len(vector)}")
    print("\n--- FIRST 10 VALUES OF EMBEDDING ---\n")
    print(vector[:10])