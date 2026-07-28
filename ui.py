"""Streamlit chat UI with model switching and streamed responses.

Run: streamlit run ui.py
"""
import ollama
import streamlit as st

from app.chat import ChatSession
from app.config import OLLAMA_HOST, OLLAMA_MODEL

st.set_page_config(page_title="Web Dev RAG Chatbot", page_icon="💬")


@st.cache_data(ttl=30)
def list_available_models() -> list[str]:
    """Queries Ollama directly (not hardcoded) so the dropdown always reflects
    whatever models are actually installed on this machine."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        names = [m["model"] for m in client.list()["models"]]
        return sorted(names) or [OLLAMA_MODEL]
    except Exception:
        return [OLLAMA_MODEL]


if "session" not in st.session_state:
    st.session_state.session = ChatSession()
    st.session_state.messages = []  # [{"role", "content", "model", "used_tool", "sources"}]

with st.sidebar:
    st.header("Web Dev RAG Chatbot")
    st.markdown(
        "Ask about **HTML/CSS**, **JavaScript**, **React**, **Node.js**, or "
        "**Drupal** — answered from a curated knowledge base. Mentioning "
        "**weather** and a city (e.g. \"what's the weather in London?\") "
        "triggers a live lookup automatically, no special command needed."
    )
    st.divider()
    models = list_available_models()
    selected_model = st.selectbox("Model", models, index=models.index(st.session_state.session.model) if st.session_state.session.model in models else 0)
    if selected_model != st.session_state.session.model:
        st.session_state.session.set_model(selected_model)
    st.caption(
        "Switching models keeps this conversation's history — only the "
        "model answering the *next* message changes. Smaller models (e.g. "
        "qwen2.5:0.5b) are much faster but noticeably less reliable at "
        "understanding what you're asking than larger ones (e.g. phi3)."
    )
    if st.button("Clear conversation"):
        st.session_state.session = ChatSession(model=st.session_state.session.model)
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            meta = [f"model: {msg['model']}"]
            if msg["used_tool"]:
                meta.append("used external tool")
            if msg["sources"]:
                meta.append("sources: " + ", ".join(msg["sources"]))
            st.caption(" | ".join(meta))

user_message = st.chat_input("Ask about HTML/CSS, JavaScript, React, Node.js, Drupal... or weather in a city")
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        session = st.session_state.session
        for text_so_far in session.ask_stream(user_message):
            placeholder.markdown(text_so_far)
        answer = session.last_answer
        placeholder.markdown(answer.answer)
        meta = [f"model: {session.model}"]
        if answer.used_tool:
            meta.append("used external tool")
        if answer.sources:
            meta.append("sources: " + ", ".join(answer.sources))
        st.caption(" | ".join(meta))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.answer,
        "model": session.model,
        "used_tool": answer.used_tool,
        "sources": answer.sources,
    })
