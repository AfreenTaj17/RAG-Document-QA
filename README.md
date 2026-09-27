# 📚 RAG-Based Document Q&A

A Retrieval-Augmented Generation based Document Question Answering system using LangChain, FAISS, Hugging Face embeddings and OpenAI.

## 🚀 Live Demo

👉 Try the RAG Document Q&A App

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rag-document-qsan.streamlit.app/)

## ✨ Features

- Semantic document retrieval using FAISS
- Hugging Face sentence-transformer embeddings
- Configurable number of retrieved documents
- Page-level source context display
- Streamlit interactive interface

The system combines **document processing, Hugging Face embeddings, FAISS vector search, LangChain, and OpenAI** to retrieve relevant information from documents before generating an answer.

## Overview

Traditional LLM-based question answering can produce incorrect or hallucinated answers when the required information is not present in the model's knowledge.

This project uses a **Retrieval-Augmented Generation (RAG)** approach:

1. Documents are loaded and processed.
2. Documents are split into smaller text chunks.
3. Text chunks are converted into vector embeddings using a Hugging Face Sentence Transformer model.
4. The embeddings are stored in a FAISS vector database.
5. When a user asks a question, the system performs semantic similarity search.
6. The most relevant document chunks are retrieved.
7. The retrieved context is provided to an OpenAI LLM.
8. The LLM generates an answer based on the retrieved context.

## Architecture

```text
                Documents
                    │
                    ▼
          Document Loading
                    │
                    ▼
             Text Splitting
                    │
                    ▼
        Hugging Face Embeddings
                    │
                    ▼
             FAISS Vector Store
                    │
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   │
    User Question             │
          │                   │
          ▼                   │
    Query Embedding           │
          │                   │
          ▼                   │
    Similarity Search ────────┘
          │
          ▼
   Relevant Context
          │
          ▼
      OpenAI LLM
          │
          ▼
    Generated Answer
          │
          ▼
    Streamlit Interface
```

## Key Features

* Document-based question answering
* Retrieval-Augmented Generation (RAG)
* Semantic document search
* Hugging Face Sentence Transformer embeddings
* FAISS vector similarity search
* LangChain-based RAG pipeline
* OpenAI LLM integration
* Streamlit web interface
* Environment variable based API-key management
* MLflow experiment tracking
* Modular project structure

## Tech Stack

| Technology                         | Purpose                              |
| ---------------------------------- | ------------------------------------ |
| Python                             | Core programming language            |
| LangChain                          | RAG pipeline and LLM orchestration   |
| Hugging Face Sentence Transformers | Text embeddings                      |
| FAISS                              | Vector storage and similarity search |
| OpenAI API                         | Large Language Model                 |
| Streamlit                          | Web application interface            |
| MLflow                             | Experiment tracking                  |
| Jupyter Notebook                   | Development and experimentation      |
| python-dotenv                      | Environment variable management      |

## Project Structure

```text
RAG-Based-Document-QA/
│
├── data/
│   └── Documents used for the QA system
│
├── notebook/
│   └── Jupyter notebooks for experimentation and development
│
├── src/
│   └── Source code for document processing and RAG components
│
├── vectorstore/
│   └── Generated FAISS vector store
│
├── app.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Files and directories excluded from Git
│
├── .env
│   └── Local environment variables and API keys
│
└── README.md
    └── Project documentation
```

> **Note:** The `.env`, `venv`, generated vector store, and local MLflow database are excluded from version control.

## How the RAG Pipeline Works

### 1. Document Loading

The system first loads the source documents from the data directory.

### 2. Text Splitting

Large documents are divided into smaller chunks so that relevant sections can be efficiently retrieved during question answering.

### 3. Embedding Generation

Each text chunk is converted into a numerical vector using a Hugging Face Sentence Transformer embedding model.

These embeddings capture the semantic meaning of the text.

### 4. Vector Storage

The generated embeddings are stored in a **FAISS vector store**, which enables efficient similarity-based retrieval.

### 5. Query Processing

When a user enters a question, the question is converted into an embedding using the same embedding model.

### 6. Similarity Search

FAISS compares the query embedding with document embeddings and retrieves the most semantically relevant chunks.

### 7. Context-Augmented Generation

The retrieved chunks are passed as context to the OpenAI LLM along with the user's question.

The model then generates an answer using the retrieved information.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Based-Document-QA.git
cd RAG-Based-Document-QA
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Do not commit the `.env` file to GitHub.

## Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in the browser and provide an interface for asking questions about the loaded documents.

## Example Workflow

```text
Upload / Load Document
        ↓
Extract Text
        ↓
Split Into Chunks
        ↓
Generate Embeddings
        ↓
Store in FAISS
        ↓
Ask a Question
        ↓
Retrieve Relevant Chunks
        ↓
Send Context + Question to LLM
        ↓
Generate Answer
```

## Experiment Tracking

MLflow is used to track experiments and monitor relevant information during development.

This helps maintain visibility into different experiments and provides a foundation for evaluating and improving the RAG pipeline.

## Advantages of the Approach

* Provides answers based on retrieved document context
* Reduces dependence on the LLM's pre-trained knowledge
* Enables semantic rather than only keyword-based search
* Allows domain-specific documents to be queried
* Separates document retrieval from answer generation
* Can be extended to support multiple document formats and data sources

## Future Improvements

* Support for additional document formats
* Improved chunking and retrieval strategies
* Hybrid keyword + semantic search
* Reranking retrieved documents
* Source citation in generated answers
* Conversation history and multi-turn QA
* RAG evaluation using retrieval and generation metrics
* Deployment using cloud infrastructure
* Authentication and multi-user support

## Author

**Afreen Taj**

* GitHub: [AfreenTaj17](https://github.com/AfreenTaj17)
* LinkedIn: [Afreen Taj](https://www.linkedin.com/in/afreentaj17/)
* Email: [tafreen173@gmail.com](mailto:tafreen173@gmail.com)

---

## License

This project is intended for educational and portfolio purposes.
