# RAG-PDF 2.0

A conversational RAG (Retrieval-Augmented Generation) agent that lets you chat with PDF documents via a Streamlit web interface. Ask questions about your document and get answers with **verbatim source quotes and page references** - or just have a general conversation.

---

## Features

- **MultiQuery RAG** - generates multiple query phrasings per question for broader, more accurate retrieval
- **Intent routing** - casual messages go directly to the LLM; document questions are routed through the full RAG pipeline
- **Verbatim citations** - answers include exact quoted passages and page numbers from the source PDF
- **Inline PDF viewer** - view the document alongside the chat (toggle-ready)
- **FAISS vector index** - fast local similarity search over document chunks

---

## Tech Stack

| Component | Library / Model |
|---|---|
| LLM | `openai/gpt-oss-120b` via [Groq](https://console.groq.com) |
| Embeddings | `gemini-embedding-2-preview` via Google AI |
| Vector store | FAISS (local) |
| Document loader | PyMuPDF |
| Framework | LangChain LCEL + Streamlit |

---

## Installation

```bash
pip install streamlit langchain-core langchain-community \
            langchain-google-genai langchain-text-splitters \
            langchain-groq faiss-cpu pymupdf
```

---

## Configuration

Create a `.streamlit/secrets.toml` file in the project root:

```toml
GOOGLE_API_KEY = "your-google-ai-api-key"
GROQ_API_KEY   = "your-groq-api-key"
```

> Get your Google AI key at [aistudio.google.com](https://aistudio.google.com/app/apikey)  
> Get your Groq key at [console.groq.com](https://console.groq.com)

---

## Usage

```bash
streamlit run RAG-PDF-3.0.py
```

1. Upload a PDF using the sidebar
2. Wait for indexing to complete
3. Ask questions about the document — or just chat

---

## Project Structure

```
.
├── RAG-PDF-3.0.py          # Main application
└── .streamlit/
    └── secrets.toml        # API keys (do not commit)
```

---

## .gitignore

Make sure to exclude your secrets:

```
.streamlit/secrets.toml
```
