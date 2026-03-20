import base64
import os
import re
import tempfile
from typing import List

import streamlit as st

# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document

except ImportError as exc:
    st.error(
        f"❌ Missing library: **{exc}**\n\n"
        "Run:\n"
        "```\npip install streamlit langchain-core langchain-community "
        "langchain-google-genai langchain-text-splitters langchain-groq "
        "faiss-cpu pymupdf\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# API Keys — .streamlit/secrets.toml
# ---------------------------------------------------------------------------
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "🔑 **GOOGLE_API_KEY not found.**\n\n"
        "Add it to `.streamlit/secrets.toml`:\n"
        "```toml\nGOOGLE_API_KEY = \"YOUR_KEY\"\n```"
    )
    st.stop()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "🔑 **GROQ_API_KEY not found.**\n\n"
        "Add it to `.streamlit/secrets.toml`:\n"
        "```toml\nGROQ_API_KEY = \"YOUR_KEY\"\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROQ_MODEL    = "openai/gpt-oss-120b"   # Groq-hosted OpenAI-compatible model
EMBED_MODEL   = "gemini-embedding-2-preview" # Google embedding model
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
TOP_K         = 4
MQ_COUNT      = 3   # number of query paraphrases

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
for k, v in {
    "messages":    [],
    "pdf_bytes":   None,
    "file_name":   None,
    "target_page": 1,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# LLM factory  — ChatGroq returns AIMessage; StrOutputParser extracts .content
# ---------------------------------------------------------------------------
def make_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_api_key,
        temperature=temperature,
    )

# ---------------------------------------------------------------------------
# Vector store  (Gemini embeddings, FAISS index)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔍 Indexing PDF…")
def build_vector_store(file_bytes: bytes, file_name: str) -> FAISS:
    """Write upload → temp file → load → chunk → embed → FAISS."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyMuPDFLoader(tmp_path)
        docs   = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=google_api_key
    )
    return FAISS.from_documents(chunks, embeddings)

# ---------------------------------------------------------------------------
# MultiQuery retrieval
# ---------------------------------------------------------------------------
_MQ_PROMPT = PromptTemplate.from_template(
    """You are an AI assistant improving document retrieval.
Generate {n} different phrasings of the question below to maximise search coverage.
Output ONLY the questions — one per line — no numbering, no extra text.

Original question: {question}

Rephrased questions:"""
)


def generate_multi_queries(question: str, llm: ChatGroq, n: int = MQ_COUNT) -> List[str]:
    """Return `n` LLM-generated rephrasings + the original question."""
    chain = _MQ_PROMPT | llm | StrOutputParser()
    raw: str = chain.invoke({"question": question, "n": n})
    extras = [q.strip() for q in raw.strip().splitlines() if q.strip()]
    seen, result = {question.lower()}, [question]
    for q in extras:
        if q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)
    return result


def multiquery_retrieve(
    question: str, vector_store: FAISS, llm: ChatGroq
) -> List[Document]:
    """Run all query variants through the retriever and deduplicate results."""
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    queries   = generate_multi_queries(question, llm)
    seen_ids: set       = set()
    all_docs: List[Document] = []
    for q in queries:
        for doc in retriever.invoke(q):
            doc_id = hash(doc.page_content)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)
    return all_docs

# ---------------------------------------------------------------------------
# RAG answer chain (LCEL)
# ---------------------------------------------------------------------------
_RAG_PROMPT = PromptTemplate.from_template(
    """You are a precise, friendly document assistant.

INSTRUCTIONS:
1. Answer the user's question using ONLY the context chunks below.
2. For every factual claim, immediately follow it with the VERBATIM source text
   in a Markdown block-quote, then the page number like this:
       > "…exact sentence(s) copied word-for-word from the document…"
       *(Page N)*
3. If multiple chunks support a claim, cite each separately.
4. If the context does not contain the answer, say so honestly — do NOT fabricate.
5. Keep your tone warm and conversational.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (include verbatim quotes and page numbers as instructed):"""
)


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        raw_page = doc.metadata.get("page", "?")
        try:
            page_display = int(raw_page) + 1   # PyMuPDF is 0-indexed
        except (ValueError, TypeError):
            page_display = raw_page
        parts.append(f"[Page {page_display}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_answer(question: str, vector_store: FAISS) -> dict:
    """Full MultiQuery RAG pipeline → {'answer': str, 'pages': list[int]}."""
    llm         = make_llm(temperature=0.2)
    docs        = multiquery_retrieve(question, vector_store, llm)
    context_str = format_docs(docs)
    chain       = _RAG_PROMPT | llm | StrOutputParser()
    answer      = chain.invoke({"context": context_str, "question": question})
    return {"answer": answer, "pages": extract_page_numbers(answer)}

# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------
_INTENT_PROMPT = PromptTemplate.from_template(
    """Classify the user message. Reply with exactly one word: RAG or CHAT — nothing else.

RAG  — the message is a question or request that needs information from an
       uploaded document to answer well.
CHAT — casual conversation, greetings, or general knowledge requiring no document.

User message: {message}

Classification:"""
)


@st.cache_data(show_spinner=False)
def classify_intent(message: str) -> str:
    llm    = make_llm(temperature=0.0)
    chain  = _INTENT_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"message": message}).strip().upper()
    return "RAG" if "RAG" in result else "CHAT"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def extract_page_numbers(text: str) -> List[int]:
    """Extract all *(Page N)* / (Page N) markers from the model's answer."""
    return sorted({
        int(n)
        for n in re.findall(r"\*?\(?\s*[Pp]age\s+(\d+)\s*\)?\*?", text)
    })


def render_pdf_viewer(pdf_bytes: bytes, page: int = 1) -> None:
    """Render PDF as a base64-embedded iframe; #page=N scrolls to that page."""
    b64 = base64.b64encode(pdf_bytes).decode()
    src = f"data:application/pdf;base64,{b64}#page={page}"
    st.components.v1.html(
        f'<iframe src="{src}" width="100%" height="780" '
        'style="border:1px solid #ddd; border-radius:6px;"></iframe>',
        height=790,
        scrolling=False,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        raw_bytes = uploaded_file.getvalue()
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.pdf_bytes   = raw_bytes
            st.session_state.file_name   = uploaded_file.name
            st.session_state.messages    = []
            st.session_state.target_page = 1
        st.success(f"✅ **{uploaded_file.name}**")

    st.divider()
    st.caption(f"RAG-PDF 3.0 · {GROQ_MODEL} via Groq · Gemini embeddings")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
chat_col = st.container()

# ── Chat (single column layout) ────────────────────────────────────────────
with chat_col:
    st.subheader("💬 Chat")

    if not st.session_state.pdf_bytes:
        st.info("⬆️ Upload a PDF in the sidebar to get started.")
        st.stop()

    # Build / retrieve cached vector store
    vector_store = build_vector_store(
        st.session_state.pdf_bytes, st.session_state.file_name
    )

    # Welcome message on first load
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Document indexed! Ask me anything about it — or just say hi 👋"
            )

    # ── Render chat history ────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── New user input ─────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask about the document, or just chat…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                intent = classify_intent(user_input)

                if intent == "RAG":
                    result = build_rag_answer(user_input, vector_store)
                    answer = result["answer"]
                else:
                    # Plain conversational reply — no document lookup
                    llm    = make_llm(temperature=0.7)
                    answer = (llm | StrOutputParser()).invoke(user_input)

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )