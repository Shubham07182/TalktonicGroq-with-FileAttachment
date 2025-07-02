import streamlit as st
import pytesseract
from datetime import datetime
import re
import requests
import pdfplumber
from PIL import Image
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import pickle
from dotenv import load_dotenv
import os


load_dotenv(".env")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_relevant_chunks(query, vector_store, top_k=3):
    query_vec = embed_model.encode([query])[0]
    similarities = []
    for item in vector_store:
        score = cosine_similarity([query_vec], [item["vector"]])[0][0]
        similarities.append((score, item["chunk"]))
    top_chunks = sorted(similarities, reverse=True)[:top_k]
    return "\n".join(chunk for _, chunk in top_chunks)
st.set_page_config(page_title="TalkTonic", layout="centered")


def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    image = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text + "\n"
            return text or "No text found, even with OCR."
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    elif uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    return "Unsupported file type."

def call_groq_model(message, model="llama3-8b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    api_key = os.getenv("GROQ_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"General error: {e}"


def get_file_hash(file):
    content = file.read()
    hash_val = hashlib.md5(content).hexdigest()
    file.seek(0)  
    return hash_val


def strip_html_tags(text):
    return re.sub(r'<[^>]*>', '', text)

def get_theme_colors(theme):
    themes = {
        "Light": {"chat_bg": "#f0f0f0", "user_bg": "#4caf50", "user_color": "white",
                  "bot_bg": "#d3d3d3", "bot_color": "black", "clear_btn_bg": "#f44336", "clear_btn_color": "white"},
        "Midnight": {"chat_bg": "#0b0c10", "user_bg": "#66fcf1", "user_color": "#0b0c10",
                     "bot_bg": "#1f2833", "bot_color": "#c5c6c7", "clear_btn_bg": "#45a29e", "clear_btn_color": "#0b0c10"},
        "Dark": {"chat_bg": "#1f1f1f", "user_bg": "#4caf50", "user_color": "white",
                 "bot_bg": "#333", "bot_color": "#f1f1f1", "clear_btn_bg": "#f44336", "clear_btn_color": "white"}
    }
    return themes.get(theme, themes["Dark"])


if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
for key, default in {
    "last_file_hash": None,
    "extracted_text": "",
    "messages": [],
    "pending_input": "",
    "theme": "Dark",
    "chunks": [],
    "embeddings": [],
    "vector_store": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


uploaded_file = st.file_uploader("Upload a PDF, Image, or Text File", type=["pdf", "png", "jpg", "jpeg", "txt"])

if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    uploaded_file.seek(0)  
    if file_hash != st.session_state.last_file_hash:
        st.session_state.extracted_text = extract_text_from_file(uploaded_file)
        st.session_state.last_file_hash = file_hash
        st.success("✅ Text extracted from the new file.")
        st.write(f"📄 Current File: `{uploaded_file.name}`")
        def chunk_text(text, chunk_size=500, overlap=100):
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
            return chunks

        st.session_state.chunks = chunk_text(st.session_state.extracted_text)
        embeddings = embed_model.encode(st.session_state.chunks)
        st.session_state.embeddings = embeddings

        st.session_state.vector_store = [
            {"chunk": chunk, "vector": vector}
            for chunk, vector in zip(st.session_state.chunks, embeddings)
        ]

        with open("vector_store.pkl", "wb") as f:
            pickle.dump(st.session_state.vector_store, f)
else:
    st.session_state.last_file_hash = None
    st.session_state.extracted_text = ""


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select the Option", ["Chat", "About"])

st.sidebar.subheader("Theme")
st.session_state.theme = st.sidebar.selectbox(
    "Choose Theme", ["Dark", "Light", "Midnight"],
    index=["Dark", "Light", "Midnight"].index(st.session_state.theme)
)
if os.path.exists("vector_store.pkl"):
    try:
        with open("vector_store.pkl", "rb") as f:
            st.session_state.vector_store = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        st.warning("⚠️ vector_store.pkl is corrupted. Starting fresh.")
        st.session_state.vector_store = []
        os.remove("vector_store.pkl")
        st.info("🗑️ Corrupted pickle file deleted.")
else:
    st.session_state.vector_store = []

if page == "Chat":
    colors = get_theme_colors(st.session_state.theme)
    st.markdown(f"""
    <style>
    .chat-container {{height: 400px; overflow-y: auto; border: 1px solid #444; padding: 10px;
        border-radius: 10px; background-color: {colors['chat_bg']}; color: {colors['bot_color']};}}
    .user-message {{background-color: {colors['user_bg']}; color: {colors['user_color']};
        padding: 8px 12px; border-radius: 10px; margin-bottom: 8px; max-width: 70%;
        float: right; clear: both; font-size: 15px;}}
    .bot-message {{background-color: {colors['bot_bg']}; color: {colors['bot_color']};
        padding: 8px 12px; border-radius: 10px; margin-bottom: 8px; max-width: 70%;
        float: left; clear: both; font-size: 15px;}}
    small {{display: block; font-size: 11px; opacity: 0.7; margin-top: 3px;}}
    </style>
    """, unsafe_allow_html=True)

   
    with st.container():
        col1, col2, col3 = st.columns([1, 5, 4])
        with col1:
            st.markdown("<div style='margin-top: 10px; font-size: 40px;'>🤖</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<h1 style='margin-bottom: 0;'>TalkTonic</h1>", unsafe_allow_html=True)
        with col3:
            current_time = datetime.now().strftime("%b %d, %Y - %I:%M %p")
            st.markdown(f"""
                <div style='text-align: right; font-size:13px; margin-top: 10px;'>
                    Theme: <b>{st.session_state.theme}</b><br>
                    <span style='color: #33ff77;'>🟢</span> Bot Status: <b>Online</b><br>
                    {current_time}
                </div>
            """, unsafe_allow_html=True)

   
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_input = ""
    with col2:
        if st.session_state.messages:
            clean_chat = "\n".join(
                f"{sender.upper()}: {strip_html_tags(msg)}"
                for sender, msg in st.session_state.messages
            )
            st.download_button("💾 Download Chat", clean_chat, file_name="talktonic_chat.txt", use_container_width=True)

   
    if st.session_state.extracted_text:
        modified_text = st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
        if st.button("Send to Bot"):
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.messages.append(("user", f"{modified_text}<small>{timestamp}</small>"))
            bot_reply = call_groq_model(modified_text)
            st.session_state.messages.append(("bot", f"{bot_reply}<small>{timestamp}</small>"))

  
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.pending_input = user_input.strip()

    if st.session_state.pending_input:
        timestamp = datetime.now().strftime("%H:%M")
        user_input = st.session_state.pending_input

        if st.session_state.vector_store:
            retrieved_context = retrieve_relevant_chunks(user_input, st.session_state.vector_store)
            prompt = f"Context:\n{retrieved_context}\n\nUser Query: {user_input}\n\nAnswer:"
        else:
            prompt = user_input

        bot_reply = call_groq_model(prompt)

        st.session_state.messages.append(("user", f"{user_input}<small>{timestamp}</small>"))
        st.session_state.messages.append(("bot", f"{bot_reply}<small>{timestamp}</small>"))
        st.session_state.pending_input = ""

    
    chat_html = """<div id="chatbox" class="chat-container">"""
    for sender, msg in st.session_state.messages:
        chat_html += f'<div class="{sender}-message">{msg}</div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)
    st.markdown("""
    <script>
    var chatbox = document.getElementById("chatbox");
    if(chatbox){chatbox.scrollTop = chatbox.scrollHeight;}
    </script>
    """, unsafe_allow_html=True)

elif page == "About":
    st.subheader("About TalkTonic")
    st.markdown("""
    *TalkTonic* is a modern chatbot UI built with Streamlit.  
    It supports multiple themes, chat history, and smart auto-scroll.

    - Built for students, devs, and creators  
    - Easily integrates with GPT models or custom AI logic  
    - Fully customizable with themes, avatars, and more  
    """)
