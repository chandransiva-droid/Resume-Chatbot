import html
import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ABOUT_ME_PATH = Path(__file__).parent / "about_me.txt"
NAME = "Chandran Siva"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
TOP_K = 3
WELCOME_MESSAGE = (
    f"Hi! I'm here to answer questions about **{NAME}**. "
    "Ask about hobbies, technologies, languages, or anything else from the bio. "
    "If it isn't in the notes, I'll tell you I don't have that information."
)
EXAMPLE_QUESTIONS = [
    "What are your hobbies?",
    "What technologies do you know?",
    "What languages do you speak?",
]

CUSTOM_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap");

#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stHeader"], .stDeployButton,
.stAppDeployButton, [data-testid="stMainMenu"] {
    display: none !important;
    visibility: hidden !important;
}

html, body, [class*="css"], .stApp, .stMarkdown, .stChatMessage {
    font-family: "DM Sans", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(99, 102, 241, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 24%),
        #f3f5fb;
}

.block-container {
    padding-top: 3.25rem;
    padding-bottom: 7rem;
    max-width: 820px;
}

h1 {
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    font-size: 2.1rem !important;
}

.hero-subtitle {
    margin-top: -0.55rem;
    margin-bottom: 1.6rem;
    color: #64748b;
    font-size: 1.02rem;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #e6ebf5;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    padding: 0.35rem 0.2rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #eef2ff;
    border-color: #dbe3ff;
}

[data-testid="stExpander"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
}

[data-testid="stChatInput"] {
    background: transparent;
}

[data-testid="stChatInput"] textarea {
    border-radius: 16px !important;
}

.source-chunk {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin: 0.45rem 0;
    color: #334155;
    font-size: 0.92rem;
    line-height: 1.45;
}

div.stButton > button {
    border-radius: 999px;
    border: 1px solid #dbe3ff;
    background: #ffffff;
    color: #3730a3;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

div.stButton > button:hover {
    background: #eef2ff;
    border-color: #6366f1;
    color: #312e81;
}

div.stButton > button[kind="secondary"] {
    color: #475569;
    border-color: #e2e8f0;
}
</style>
"""

st.set_page_config(
    page_title=f"Ask Me Anything About {NAME}",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title(f"🤖 Ask Me Anything About {NAME}")
st.markdown(
    f'<p class="hero-subtitle">A personal Q&amp;A bot trained on notes about {NAME}.</p>',
    unsafe_allow_html=True,
)

missing_api_key = not OPENAI_API_KEY
missing_about_me = not ABOUT_ME_PATH.is_file()

if missing_api_key:
    st.error(
        "**OpenAI API key is missing.** Add `OPENAI_API_KEY` to a `.env` file "
        "in this project folder, then restart the app."
    )
if missing_about_me:
    st.error(
        "**`about_me.txt` was not found.** Put the file next to `rag_chatbot.py` "
        f"at `{ABOUT_ME_PATH.resolve()}`."
    )
if missing_api_key or missing_about_me:
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


def welcome_message() -> dict:
    return {"role": "assistant", "content": WELCOME_MESSAGE, "sources": []}


def load_about_me(path: Path = ABOUT_ME_PATH) -> str:
    if not path.is_file():
        st.error(
            "**`about_me.txt` was not found.** Put the file next to `rag_chatbot.py` "
            f"at `{path.resolve()}`."
        )
        st.stop()
    return path.read_text(encoding="utf-8").strip()


def split_into_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def make_overlapping_chunks(
    sentences: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    if not sentences:
        return []
    if len(sentences) <= chunk_size:
        return [" ".join(sentences)]

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0
    while start < len(sentences):
        chunk = sentences[start : start + chunk_size]
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(sentences):
            break
        start += step
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(query_embedding: list[float], chunk_embeddings: list[list[float]]) -> np.ndarray:
    query = np.array(query_embedding)
    chunks = np.array(chunk_embeddings)
    query_norm = np.linalg.norm(query)
    chunk_norms = np.linalg.norm(chunks, axis=1)
    return (chunks @ query) / (chunk_norms * query_norm)


def top_k_chunks(question: str, k: int = TOP_K) -> list[str]:
    question_embedding = embed_texts([question])[0]
    scores = cosine_similarity(question_embedding, st.session_state.embeddings)
    k = min(k, len(st.session_state.chunks))
    top_indices = np.argsort(scores)[::-1][:k]
    return [st.session_state.chunks[i] for i in top_indices]


def answer_from_context(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    instruction = (
        f"Answer ONLY using this context about {NAME}. "
        "If the answer is not in the context, say you don’t have that information."
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": f"{instruction}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for index, chunk in enumerate(sources, start=1):
            st.markdown(
                f'<div class="source-chunk"><strong>Chunk {index}</strong><br>{html.escape(chunk)}</div>',
                unsafe_allow_html=True,
            )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])


def initialize_knowledge_base() -> None:
    if "chunks" in st.session_state and "embeddings" in st.session_state:
        return

    with st.spinner("Preparing knowledge base..."):
        text = load_about_me()
        sentences = split_into_sentences(text)
        chunks = make_overlapping_chunks(sentences)
        if not chunks:
            st.error("about_me.txt is empty, so no knowledge base could be built.")
            st.stop()

        embeddings = embed_texts(chunks)

    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings


initialize_knowledge_base()

if "messages" not in st.session_state:
    st.session_state.messages = [welcome_message()]

example_cols = st.columns(3)
prompt = None
for column, question in zip(example_cols, EXAMPLE_QUESTIONS):
    if column.button(question, use_container_width=True, key=f"example_{question}"):
        prompt = question

clear_clicked = st.button("Clear chat", type="secondary")

for message in st.session_state.messages:
    render_message(message)

typed_prompt = st.chat_input("Ask me anything...")
if typed_prompt:
    prompt = typed_prompt

if clear_clicked:
    st.session_state.messages = [welcome_message()]
    st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved_chunks = top_k_chunks(prompt)
            reply = answer_from_context(prompt, retrieved_chunks)
        st.markdown(reply)
        render_sources(retrieved_chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": retrieved_chunks}
    )
