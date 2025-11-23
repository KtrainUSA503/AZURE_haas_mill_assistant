"""
Haas Mill Operator's Manual Assistant - Enterprise Version
Azure-ready Streamlit app using OpenAI + Pinecone + Azure Blob (managed identity)
"""

import os
import time
import json
import csv
from datetime import datetime

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Haas Mill Assistant",
    page_icon="🔧",
    layout="wide"
)

# -----------------------------------------------------------------------------
# ENVIRONMENT / CONFIG
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_5tsZi_JEAMygi9zKeJmtXQoxekZZzuUxLS1RTuMzXNgxoAq2m5gkjGoHjDNUmmWXUdg7D")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "haas-mill-manual")

# Blob settings (managed identity)
BLOB_URL = os.getenv("MANUAL_CHUNKS_BLOB_URL", "")  # e.g., https://stcnchasadev01.blob.core.windows.net/haas-mill-data/haas_mill_chunks.json

# -----------------------------------------------------------------------------
# USER ACCOUNTS (DEMO ONLY; MOVE TO SECURE STORE IN PROD)
# -----------------------------------------------------------------------------
USERS = {
    "aaron.colby": {"password": "Keith2025", "name": "Aaron Colby", "role": "Operator"},
    "kenny.yukich": {"password": "Keith2025", "name": "Kenny Yukich", "role": "Operator"},
    "brian.feigner": {"password": "Keith2025", "name": "Brian Feigner", "role": "IT Administrator"},
    "admin": {"password": "Keith2025Admin", "name": "Administrator", "role": "Admin"}
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
def log_activity(username, action, details=""):
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "action": action,
            "details": details
        }
        st.session_state.setdefault('activity_log', []).append(entry)
    except Exception:
        pass

def log_query(username, question, response, sources, feedback=None):
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "question": question,
            "response_length": len(response or ""),
            "num_sources": len(sources or []),
            "sources": [s.get('page') for s in (sources or [])],
            "feedback": feedback
        }
        st.session_state.setdefault('query_log', []).append(entry)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# SESSION / RATE LIMITS
# -----------------------------------------------------------------------------
def check_session_timeout():
    if 'last_activity' in st.session_state:
        if time.time() - st.session_state.last_activity > 1800:
            st.session_state.authenticated = False
            st.session_state.username = None
            return True
    return False

def update_activity():
    st.session_state.last_activity = time.time()

def check_rate_limit(username):
    st.session_state.setdefault('query_counts', {})
    today = datetime.now().date().isoformat()
    key = f"{username}_{today}"
    st.session_state['query_counts'].setdefault(key, 0)
    return st.session_state['query_counts'][key] < 100

def increment_query_count(username):
    st.session_state.setdefault('query_counts', {})
    today = datetime.now().date().isoformat()
    key = f"{username}_{today}"
    st.session_state['query_counts'].setdefault(key, 0)
    st.session_state['query_counts'][key] += 1

# -----------------------------------------------------------------------------
# AUTH
# -----------------------------------------------------------------------------
def authenticate_user():
    def login_submitted():
        username = st.session_state.login_username
        password = st.session_state.login_password
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_name = USERS[username]["name"]
            st.session_state.user_role = USERS[username]["role"]
            st.session_state.last_activity = time.time()
            st.session_state.login_attempts = 0
            log_activity(username, "login", "Successful login")
        else:
            st.session_state.authenticated = False
            st.session_state.login_attempts = st.session_state.get('login_attempts', 0) + 1
            log_activity(username, "login_failed", "Failed login attempt")

    if check_session_timeout():
        st.warning("⏰ Your session has expired for security. Please log in again.")
        log_activity(st.session_state.get('username', 'unknown'), "timeout", "Session timeout")

    if not st.session_state.get("authenticated", False):
        st.title("🔧 Keith Manufacturing - Machine Assistant")
        st.markdown("### Secure Login")
        st.info("👋 Welcome to the Haas Mill Manual Assistant")

        with st.form("login_form"):
            st.text_input("Username:", key="login_username")
            st.text_input("Password:", type="password", key="login_password")
            st.form_submit_button("Login", on_click=login_submitted)

        if st.session_state.get('login_attempts', 0) > 0:
            st.error("❌ Invalid username or password. Please try again.")

        if st.session_state.get('login_attempts', 0) >= 3:
            st.warning("⚠️ Multiple failed login attempts. Contact IT for help: brian@keithmfg.com")
        st.caption("For access, contact your supervisor or IT department")
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
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ I Understand and Agree", use_container_width=True):
                st.session_state.disclaimer_accepted = True
                log_activity(st.session_state.username, "disclaimer_accepted", "User acknowledged safety disclaimer")
                st.rerun()
            if st.button("❌ I Do Not Agree", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.disclaimer_accepted = False
                st.info("You must accept the safety disclaimer to use this system.")
                st.stop()
        st.stop()

# -----------------------------------------------------------------------------
# CLIENTS
# -----------------------------------------------------------------------------
@st.cache_resource
def init_clients():
    try:
        if not OPENAI_API_KEY or not PINECONE_API_KEY:
            st.error("Missing API keys. Set OPENAI_API_KEY and PINECONE_API_KEY in App Settings.")
            st.stop()
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        return openai_client, index
    except Exception as e:
        st.error(f"Failed to initialize API clients: {str(e)}")
        st.info("Please contact IT support: brian@keithmfg.com")
        st.stop()

# -----------------------------------------------------------------------------
# DATA LOADER (AZURE BLOB VIA MANAGED IDENTITY)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_manual_chunks_from_blob():
    if not BLOB_URL:
        st.error("MANUAL_CHUNKS_BLOB_URL not set. Configure the blob URL in App Settings.")
        st.stop()
    try:
        credential = DefaultAzureCredential()
        blob_client = BlobClient.from_blob_url(BLOB_URL, credential=credential)
        stream = blob_client.download_blob()
        data = stream.readall()
        chunks = json.loads(data)
        return chunks
    except Exception as e:
        st.error("Could not load manual chunks from Azure Blob Storage.")
        st.error(f"Details: {str(e)}")
        st.info("Ensure App Service managed identity has 'Storage Blob Data Reader' on the storage account.")
        st.stop()

# -----------------------------------------------------------------------------
# RAG FUNCTIONS
# -----------------------------------------------------------------------------
def get_query_embedding(query, client):
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
    return resp.data[0].embedding

def search_manual(query, index, client, top_k=5):
    query_vec = get_query_embedding(query, client)
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    return results

def generate_response(query, context_chunks, client):
    context = "\n\n---\n\n".join([
        f"[Page {m['metadata']['page']}]\n{m['metadata']['text']}" for m in context_chunks
    ])
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
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert Haas Mill operator assistant. Provide clear, accurate, and safety-conscious answers based on the official operator's manual."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return chat.choices[0].message.content

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if not authenticate_user():
    st.stop()

show_safety_disclaimer()
update_activity()

st.title("🔧 Haas Mill Operator's Manual Assistant")
st.markdown("### Next Generation Control - 15\" LCD (96-8210)")
st.caption(f"🔒 Logged in as: {st.session_state.user_name} ({st.session_state.user_role})")

st.markdown("""
Ask questions about operating your Haas Mill! This assistant searches the 550-page operator's manual 
to provide accurate answers with page references.

Examples:
- How do I set up a work offset?
- What is the proper procedure for tool changes?
- How do I use the probe system?
- What do the alarm codes mean?
- How do I calibrate the machine?
""")
st.divider()

# Preload manual chunks (for optional local fallback or validation view)
manual_chunks = load_manual_chunks_from_blob()

# Chat history
st.session_state.setdefault('messages', [])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}** (Relevance: {source['score']:.2%})")
                    st.text((source['text'] or "")[:300] + "...")
                    st.divider()

if prompt := st.chat_input("Ask about the Haas Mill operation..."):
    if not check_rate_limit(st.session_state.username):
        st.error("⚠️ You've reached the daily limit of 100 questions. If you need more access, contact your supervisor.")
        st.stop()

    increment_query_count(st.session_state.username)
    update_activity()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            try:
                openai_client, index = init_clients()
                results = search_manual(prompt, index, openai_client)

                if not results.matches:
                    response = "I couldn't find relevant information in the manual for that question. Please try rephrasing, check the physical manual, or ask your supervisor."
                    sources = []
                else:
                    response = generate_response(prompt, results.matches, openai_client)
                    sources = [
                        {"page": m.metadata.get('page'),
                         "score": m.score,
                         "text": m.metadata.get('text', '')}
                        for m in results.matches
                    ]

                st.markdown(response)

                if sources:
                    with st.expander("📚 View Sources"):
                        for s in sources:
                            st.markdown(f"**Page {s['page']}** (Relevance: {s['score']:.2%})")
                            st.text((s['text'] or "")[:300] + "...")
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
                log_query(st.session_state.username, prompt, response, sources)

                st.divider()
                c1, c2, c3 = st.columns([1, 1, 3])
                with c1:
                    if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.messages)}"):
                        log_query(st.session_state.username, prompt, response, sources, feedback="positive")
                        st.success("Thanks for the feedback!")
                with c2:
                    if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.messages)}"):
                        log_query(st.session_state.username, prompt, response, sources, feedback="negative")
                        st.success("Thanks for the feedback! We'll work to improve.")
            except Exception as e:
                st.error("⚠️ An error occurred while processing your question.")
                st.error(f"Error details: {str(e)}")
                st.info("Try rephrasing, and if the problem persists, contact IT support: brian@keithmfg.com")
                log_activity(st.session_state.username, "error", f"Error processing query: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    today = datetime.now().date().isoformat()
    user_key = f"{st.session_state.username}_{today}"
    queries_today = st.session_state.get('query_counts', {}).get(user_key, 0)
    st.metric("Your Questions Today", f"{queries_today}/100")

    estimated_cost = queries_today * 0.024
    st.metric("Estimated Cost Today", f"${estimated_cost:.2f}")
    st.caption(f"Last activity: {datetime.fromtimestamp(st.session_state.last_activity).strftime('%H:%M:%S')}")
    st.divider()

    st.header("About")
    st.markdown("""
    This assistant uses RAG to answer questions about the Haas Mill Operator's Manual (Revision U, December 2024).
    - Searches 550 pages
    - Provides answers with page references
    - Cites official sources
    """)
    st.divider()

    st.header("⚠️ Safety Reminder")
    st.error("Always verify critical operations in the physical manual.")
    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔓 Logout"):
        log_activity(st.session_state.username, "logout", "User logged out")
        st.session_state.authenticated = False
        st.session_state.disclaimer_accepted = False
        st.session_state.username = None
        st.rerun()

    st.divider()
    st.caption("🔒 Secure System for Keith Manufacturing")
    st.caption("Built with Streamlit • OpenAI • Pinecone • Azure Blob (MI)")
    st.caption("Version 2.0 - Enterprise Edition")

    if st.session_state.get('user_role') == 'Admin':
        st.divider()
        st.header("👨‍💼 Admin Panel")
        if st.button("View Activity Logs"):
            st.write(st.session_state.get('activity_log', []))
        if st.button("View Query Logs"):
            st.write(st.session_state.get('query_log', []))
