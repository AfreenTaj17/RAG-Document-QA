from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_pdf


def chunk_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    pdf_path = "data/data.pdf"

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    print("\n--- FIRST CHUNK ---\n")
    print(chunks[0].page_content)

    print("\n--- FIRST CHUNK METADATA ---\n")
    print(chunks[0].metadata)