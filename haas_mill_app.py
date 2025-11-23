"""
Haas Mill Operator's Manual Assistant - Azure-friendly, low-cost
Streamlit + OpenAI + Azure Blob (managed identity) + local vector search (JSON)
"""

import os
import time
import json
from datetime import datetime

import numpy as np
import streamlit as st
from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Haas Mill Assistant", page_icon="🔧", layout="wide")

# -----------------------------------------------------------------------------
# ENVIRONMENT / CONFIG
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set in App Service > Configuration
BLOB_URL = os.getenv("MANUAL_CHUNKS_BLOB_URL")  # e.g. https://<acct>.blob.core.windows.net/<container>/haas_mill_chunks.json

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # low-cost, adjust if you prefer gpt-4-turbo

TOP_K = 5  # number of chunks to retrieve

# -----------------------------------------------------------------------------
# SIMPLE AUTH (demo)
# -----------------------------------------------------------------------------
USERS = {
    # Consider removing this and using Azure App Service Authentication later
    "admin": {"password": os.getenv("APP_ADMIN_PASSWORD", ""), "name": "Administrator", "role": "Admin"}
}

def authenticate_user():
    def login_submitted():
        username = st.session_state.login_username
        password = st.session_state.login_password
        if username in USERS and USERS[username]["password"] and USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_name = USERS[username]["name"]
            st.session_state.user_role = USERS[username]["role"]
            st.session_state.last_activity = time.time()
        else:
            st.session_state.authenticated = False
            st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1

    if not st.session_state.get("authenticated", False):
        st.title("🔧 Keith Manufacturing - Machine Assistant")
        st.markdown("### Secure Login")
        with st.form("login_form"):
            st.text_input("Username:", key="login_username")
            st.text_input("Password:", type="password", key="login_password")
            st.form_submit_button("Login", on_click=login_submitted)
        if st.session_state.get('login_attempts', 0) > 0:
            st.error("❌ Invalid username or password. Please try again.")
        return False
    return True

# -----------------------------------------------------------------------------
# SAFETY DISCLAIMER
# -----------------------------------------------------------------------------
def show_safety_disclaimer():
    if 'disclaimer_accepted' not in st.session_state:
        st.session_state.disclaimer_accepted = False
    if not st.session_state.disclaimer_accepted:
        st.title("⚠️ Important Safety Notice")
        st.error("CRITICAL: READ BEFORE USING THIS SYSTEM — This AI assistant is a REFERENCE TOOL ONLY.")
        st.warning(
            "Safety Requirements:\n"
            "• Always verify critical procedures in the physical operator's manual\n"
            "• Never rely solely on AI for safety-critical operations\n"
            "• Consult your supervisor for any unclear procedures\n"
            "• In case of emergency, follow official safety protocols\n"
            "• This tool does NOT replace proper training or certification\n\n"
            "Liability Notice:\n"
            "Information may be incomplete or outdated. Use at your own risk."
        )
        if st.button("✅ I Understand and Agree", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.experimental_rerun()
        st.stop()

# -----------------------------------------------------------------------------
# CLIENTS
# -----------------------------------------------------------------------------
@st.cache_resource
def init_openai():
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY. Set it in App Service > Configuration > Application settings.")
        st.stop()
    return OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# LOAD CHUNKS FROM AZURE BLOB (MANAGED IDENTITY)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_chunks():
    if not BLOB_URL:
        st.error("MANUAL_CHUNKS_BLOB_URL not set. Configure the blob URL in App Service settings.")
        st.stop()
    try:
        credential = DefaultAzureCredential()  # uses App Service Managed Identity
        blob_client = BlobClient.from_blob_url(BLOB_URL, credential=credential)
        stream = blob_client.download_blob()
        data = stream.readall()
        chunks = json.loads(data)
        # Expect each chunk like: {"page": 123, "text": "..."}
        assert isinstance(chunks, list) and len(chunks) > 0
        return chunks
    except Exception as e:
        st.error("Could not load manual chunks from Azure Blob Storage.")
        st.error(f"Details: {str(e)}")
        st.info("Ensure App Service managed identity has 'Storage Blob Data Reader' on the storage account.")
        st.stop()

# -----------------------------------------------------------------------------
# EMBEDDINGS FOR CHUNKS (ONE-TIME CACHE)
# -----------------------------------------------------------------------------
@st.cache_resource
def build_chunk_embeddings(client, chunks):
    # Compute embeddings for all chunks once; cached across requests
    texts = [c.get("text", "") for c in chunks]
    # Batch to avoid long requests
    embeddings = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    emb = np.array(embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norms
    return emb  # shape: (N, 1536)

def embed_query(client, query):
    resp = client.embeddings.create(model=EMBED_MODEL, input=query)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec

# -----------------------------------------------------------------------------
# SEARCH (LOCAL COSINE SIMILARITY)
# -----------------------------------------------------------------------------
def search_chunks(query_vec, chunk_emb, top_k=TOP_K):
    # cosine similarity with normalized vectors is dot product
    scores = chunk_emb @ query_vec
    idx = np.argsort(-scores)[:top_k]
    return idx, scores[idx]

# -----------------------------------------------------------------------------
# ANSWER GENERATION
# -----------------------------------------------------------------------------
def compose_context(chunks, idxs, scores):
    parts = []
    for i, score in zip(idxs, scores):
        c = chunks[int(i)]
        parts.append(f"[Page {c.get('page')}] (score: {float(score):.2f})\n{c.get('text','')}")
    return "\n\n---\n\n".join(parts)

def generate_answer(client, query, context):
    prompt = f"""You are an expert assistant for Haas Mill operators at Keith Manufacturing. Answer questions based on the official operator's manual.

CRITICAL RULES:
1. Only answer based on the manual sections provided below
2. If information isn't in the provided sections, say "I don't see that specific information in the manual sections I found. Please check the physical manual or consult your supervisor."
3. ALWAYS cite page numbers for your answers
4. For safety-critical operations, emphasize verifying with the physical manual and supervisor
5. If a question is unclear, ask for clarification

RESPONSE FORMAT:
- Start with the direct answer
- Provide step-by-step instructions if applicable
- Cite page numbers in parentheses: (page 45)
- End with relevant safety warnings if applicable
- Use clear, simple language

Manual Sections:
{context}

User Question: {query}

Answer:"""
    chat = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert Haas Mill operator assistant. Provide clear, accurate, and safety-conscious answers based on the official operator's manual."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=900
    )
    return chat.choices[0].message.content

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if not authenticate_user():
    st.stop()

show_safety_disclaimer()

client = init_openai()
chunks = load_chunks()
chunk_emb = build_chunk_embeddings(client, chunks)

st.title("🔧 Haas Mill Operator's Manual Assistant")
st.markdown('### Next Generation Control - 15" LCD (96-8210)')
st.caption(f"🔒 Logged in as: {st.session_state.get('user_name','User')} ({st.session_state.get('user_role','User')})")
st.divider()

st.session_state.setdefault('messages', [])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for s in message["sources"]:
                    st.markdown(f"**Page:** {s['page']}")
                    st.text((s['text'] or "")[:300] + "...")
                    st.divider()

if prompt := st.chat_input("Ask about the Haas Mill operation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            try:
                qvec = embed_query(client, prompt)
                idxs, scores = search_chunks(qvec, chunk_emb, TOP_K)
                context = compose_context(chunks, idxs, scores)
                response = generate_answer(client, prompt, context)
                sources = [{"page": chunks[int(i)].get("page"), "text": chunks[int(i)].get("text","")} for i in idxs]
                st.markdown(response)
                with st.expander("📚 View Sources"):
                    for s in sources:
                        st.markdown(f"**Page {s['page']}**")
                        st.text((s['text'] or "")[:300] + "...")
                        st.divider()
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
            except Exception as e:
                st.error("⚠️ An error occurred while processing your question.")
                st.error(f"Error details: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    st.caption(f"Chunks loaded: {len(chunks)}")
    st.caption("Retrieval: Local cosine similarity (no Pinecone)")
    st.caption("Storage: Azure Blob (managed identity)")
    st.caption("Model: " + CHAT_MODEL)
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()
    if st.button("🔓 Logout"):
        st.session_state.authenticated = False
        st.session_state.disclaimer_accepted = False
        st.session_state.username = None
        st.experimental_rerun()