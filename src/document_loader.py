from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    print("Starting PDF loading...")
    loader = PyPDFLoader(file_path)
    print("Loader created...")
    documents = loader.load()
    print("PDF loaded successfully...")
    return documents


if __name__ == "__main__":
    pdf_path = "data/data.pdf"

    print(f"PDF path: {pdf_path}")

    documents = load_pdf(pdf_path)

    print(f"Total pages loaded: {len(documents)}")

    print("\n--- FIRST PAGE CONTENT ---\n")
    print(documents[0].page_content[:1000])

    print("\n--- FIRST PAGE METADATA ---\n")
    print(documents[0].metadata)