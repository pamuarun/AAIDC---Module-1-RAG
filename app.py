# -- coding: utf-8 --
"""
EDUBOT - Full Streamlit App with FAISS, LLM, Multi-File Upload, Styled UI, Memory, Academic Check
"""
import streamlit as st
import os, re, time
from PyPDF2 import PdfReader
from PIL import Image
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader
import evaluate
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="EDUBOT", page_icon="🤖", layout="wide")
DB_FAISS_PATH = r"D:\AAIDC\Project 1\vectorstore"
DATA_PATH = r"D:\AAIDC\Project 1\Data"

# ============================ #
# Step 2: Google API Key
# ============================ #
from dotenv import load_dotenv
import os

# Load .env file using absolute path
load_dotenv(r"D:\AAIDC\Project 1\.env")  

# Access API key
API_KEY = os.getenv("GOOGLE_API_KEY")


if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# =========================
# Load FAISS DB
# =========================
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS(embeddings.embed_query, embeddings.embed_documents, [])
    return db, db.as_retriever(search_kwargs={"k": 10}), embeddings

db, retriever, embeddings = load_retriever()

# =========================
# LLM & Prompt Template
# =========================
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0.2,
    max_output_tokens=1200
)

FULL_PROMPT_TEMPLATE = """
You are EDUBOT, an AI tutor for K-12 students. 
Act like a flexible teacher who adapts explanations to the student’s intent.

---

### Memory & Context Rules:
- Always use *chat history* to interpret vague follow-ups (e.g., "it", "this", "go with that").
- Continue the flow instead of repeating the same explanation.
- If FAISS context is weak, *fallback to general academic knowledge*.
- If the student gives acknowledgments like "okay", "yes", "continue", interpret them as *follow-up requests*.

---

### Question-Type Rules (strict):
- If the question starts with *Who* → answer only who (person, group, entity) with background, role, contributions, legacy.
- If the question starts with *What* → answer only what (definition, fact, meaning) with scope, uses, and applications.
- If the question starts with *When* → answer only time-related details.
- If the question starts with *Why* → answer only reasons/importance.
- If the question starts with *How* → answer only steps, process, or explanation.
- Do not mix categories unless the student explicitly asks.

---

### Off-Topic Rules (very strict):
- 🚫 Do *NOT* answer questions about:
  - Movies, actors, or celebrities
  - Jokes, memes, or humor requests
  - Politics, political leaders, or elections
  - Personal/private questions unrelated to academics
- Instead, politely respond: "This question is not related to your study material. Please ask me something academic."

---

### Depth Control:
- Always expand answers into *at least 4–5 lines*.
- Provide examples, applications, and relevant context.
- Avoid irrelevant details.

---

Chat History:
{chat_history}

Context from study material:
{context}

Student Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=FULL_PROMPT_TEMPLATE, input_variables=["chat_history","context","question"])
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=False,
    output_key="answer"
)

# =========================
# Evaluation
# =========================
bleu_metric = evaluate.load("bleu")
rouge_metric = Rouge()

def semantic_similarity_score(reference, generated, embed_model=embeddings):
    if not reference.strip() or not generated.strip():
        return None
    ref_vec = embed_model.embed_query(reference)
    gen_vec = embed_model.embed_query(generated)
    score = cosine_similarity([ref_vec], [gen_vec])[0][0]
    return round(score,4)

def evaluate_response(reference, generated):
    scores = {"BLEU": None, "ROUGE": None, "SemanticSim": None}
    if reference and reference.strip():
        scores["BLEU"] = bleu_metric.compute(predictions=[generated], references=[[reference]])["bleu"]
        rouge_scores = rouge_metric.get_scores(generated, reference)[0]
        scores["ROUGE"] = rouge_scores
        scores["SemanticSim"] = semantic_similarity_score(reference, generated)
    return scores

# =========================
# Academic Question Check
# =========================
def is_academic_question(question):
    followups = r"\b(ok|okay|yes|continue|go with this|that one|steps in it|the 3rd one)\b"
    if re.search(followups, question.lower()):
        return True
    non_academic_patterns = [
        r"\b(joke|funny|politics|movie|celebrity|personal)\b",
        r"\b(who|where|when) is .* president\b"
    ]
    for pat in non_academic_patterns:
        if re.search(pat, question.lower()):
            return False
    return True

# =========================
# Styling
# =========================
st.markdown("""
<style>
.stApp, .stApp > .css-18e3th9, .stApp > .block-container,
.stApp > .main > div, .stApp > div[role="main"] > div:last-child {background-color: #02213f !important; color: #02213f !important;}
header, .css-1v3fvcr, .css-18e3th9 { background-color: #02213f !important; color: #FFFFFF !important;}
[data-testid="stSidebar"] { background-color: #02213f !important; color: #FFFFFF !important; }
.sidebar-section { background-color: #0f07f7 !important; color: #FFFFFF !important; padding: 8px 12px !important;
border-radius: 8px !important; margin-bottom: 12px !important; font-weight: bold !important;
transition: transform 0.3s, box-shadow 0.3s; }
.sidebar-section:hover { transform: translateY(-3px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.stButton>button, div.stFileUploader>div>button { background-color: #FF0000 !important; color: #FFFFFF !important;
border: none !important; width: 100% !important; font-weight: bold !important;
transition: transform 0.3s, box-shadow 0.3s; }
.stButton>button:hover, div.stFileUploader>div>button:hover { transform: translateY(-3px);
box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.header-title { color: #FFFFFF !important; font-size: 60px !important; font-weight: bold !important; text-align: center !important; }
.sub-title { color: #FFFFFF !important; text-align: center !important; font-size: 24px !important; margin-bottom: 20px !important; }
.chat-bubble { display: inline-block; padding: 12px 16px; border-radius: 16px; margin: 6px 0;
max-width: 80%; word-wrap: break-word; font-size: 16px; }
.user-bubble { background-color: #0f07f7 !important; color: white !important; }
.assistant-bubble { background-color: #9b9dfa !important; color: #000 !important; }
.user-icon, .bot-icon { width:32px; height:32px; background-size: contain; display:inline-block; margin-right:8px; }
.user-icon { background-image: url('https://img.icons8.com/color/48/user.png'); }  
.bot-icon { background-image: url('https://img.icons8.com/color/48/bot.png'); }
.flex-container { display:flex; align-items:flex-start; margin-bottom:8px; }
.flex-end { justify-content:flex-end; }
.flex-start { justify-content:flex-start; }
.st-chat-input, .st-chat-input textarea, .st-chat-input div[role="textbox"] { background-color: #24142b !important;
color: #FFFFFF !important; border: 1px solid #0f07f7 !important; border-radius: 12px !important; padding: 8px !important; }
.st-chat-input button[type="submit"] { background-color: #0f07f7 !important; color: #FFFFFF !important;
border-radius: 12px !important; font-weight: bold !important; transition: transform 0.2s, box-shadow 0.2s; }
.st-chat-input button[type="submit"]:hover { background-color: #7066ff !important; transform: translateY(-2px);
box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
.st-chat-input::after { content: ""; display: block; height: 20px; background-color: #24142b !important; }
.message-history-item { background-color: rgba(167,7,247,0.2) !important; padding: 6px 8px !important;
border-radius: 6px !important; margin-bottom: 4px !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='header-title'>🤖 EduRAG - Intelligent Document Q&A</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'> 🪄 Learn Smarter, Not Harder!</div>", unsafe_allow_html=True)

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Constants / Paths
# =========================
DATA_PATH = "uploaded_files"
DB_FAISS_PATH = "faiss_db"

# Ensure folder exists
os.makedirs(DATA_PATH, exist_ok=True)

# =========================
# Sidebar (Controls Only) - Fixed with Proper Message Placement
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-section">📁 Upload Files</div>', unsafe_allow_html=True)
    
    # --- File uploader with dynamic key ---
    uploaded_files = st.file_uploader(
        "", 
        type=["txt","pdf","ppt","pptx","doc","docx","xls","xlsx"], 
        label_visibility="collapsed", 
        accept_multiple_files=True,
        key=st.session_state.get("files_uploader_key", "files_uploader")
    )

    files_msg_container = st.container()

    st.markdown('<div class="sidebar-section">🖼️ Upload Images</div>', unsafe_allow_html=True)
    
    # --- Image uploader with dynamic key ---
    uploaded_images = st.file_uploader(
        "", 
        type=["jpg","jpeg","png"], 
        label_visibility="collapsed", 
        accept_multiple_files=True,
        key=st.session_state.get("images_uploader_key", "images_uploader")
    )

    images_msg_container = st.container()

    # --- Initialize session trackers ---
    if "summarized_files" not in st.session_state:
        st.session_state.summarized_files = set()
    if "summarized_images" not in st.session_state:
        st.session_state.summarized_images = set()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files_state" not in st.session_state:
        st.session_state.uploaded_files_state = []
    if "uploaded_images_state" not in st.session_state:
        st.session_state.uploaded_images_state = []

    # --- Track uploads via session state ---
    if uploaded_files:
        st.session_state.uploaded_files_state = uploaded_files
    if uploaded_images:
        st.session_state.uploaded_images_state = uploaded_images

    # --- Display file upload messages ---
    if st.session_state.uploaded_files_state:
        with files_msg_container:
            for uploaded_file in st.session_state.uploaded_files_state:
                if uploaded_file.name in st.session_state.summarized_files:
                    st.success(f"✅ File '{uploaded_file.name}' already summarized.")
                else:
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

    # --- Display image upload messages ---
    if st.session_state.uploaded_images_state:
        with images_msg_container:
            for uploaded_image in st.session_state.uploaded_images_state:
                if uploaded_image.name in st.session_state.summarized_images:
                    st.success(f"✅ Image '{uploaded_image.name}' already processed.")
                else:
                    st.success(f"✅ Image '{uploaded_image.name}' uploaded successfully!")

    # --- Chats section ---
    st.markdown('<div class="sidebar-section">💬 Chats</div>', unsafe_allow_html=True)
    if st.button("➕ New Chat"):
        # Clear session state
        st.session_state.messages = []
        st.session_state.summarized_files = set()
        st.session_state.summarized_images = set()
        st.session_state.uploaded_files_state = []
        st.session_state.uploaded_images_state = []

        # Reset uploader widgets by changing keys
        st.session_state["files_uploader_key"] = f"files_uploader_{time.time()}"
        st.session_state["images_uploader_key"] = f"images_uploader_{time.time()}"

        files_msg_container.empty()
        images_msg_container.empty()
        st.success("Started a new chat!")

    # --- Message History section ---
    st.markdown('<div class="sidebar-section">🕘 Message History</div>', unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.messages):
        st.markdown(
            f"<div class='message-history-item'>{idx+1}. {msg['role'].capitalize()}: {msg['content']}</div>", 
            unsafe_allow_html=True
        )

    # --- Logout button ---
    if st.button("🔒 Logout"):
        # Clear session state
        st.session_state.messages = []
        st.session_state.summarized_files = set()
        st.session_state.summarized_images = set()
        st.session_state.uploaded_files_state = []
        st.session_state.uploaded_images_state = []

        # Reset uploader widgets by changing keys
        st.session_state["files_uploader_key"] = f"files_uploader_{time.time()}"
        st.session_state["images_uploader_key"] = f"images_uploader_{time.time()}"

        files_msg_container.empty()
        images_msg_container.empty()
        st.success("You have logged out successfully! 🙂")

# =========================
# Main Area: Use session state uploads
# =========================
uploaded_files = st.session_state.uploaded_files_state
uploaded_images = st.session_state.uploaded_images_state


# =========================
# Main Area: File Processing with Animated Summary
# =========================
# === FILES ===
if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if uploaded_file.name in st.session_state.summarized_files:
            continue

        text_content, summary = "", ""

        try:
            # Extract Text
            if ext == ".txt":
                text_content = open(temp_file_path, "r", encoding="utf-8").read()
            elif ext == ".pdf":
                text_content = "".join([p.extract_text() or "" for p in PdfReader(temp_file_path).pages])
            elif ext in [".doc", ".docx"]:
                text_content = "\n".join([d.page_content for d in Docx2txtLoader(temp_file_path).load()])
            elif ext in [".ppt", ".pptx"]:
                text_content = "\n".join([d.page_content for d in UnstructuredPowerPointLoader(temp_file_path).load()])
            elif ext in [".xls", ".xlsx"]:
                text_content = "\n".join([d.page_content for d in UnstructuredExcelLoader(temp_file_path).load()])

            if text_content.strip():
                # Add to FAISS DB
                db.add_texts([text_content])
                db.save_local(DB_FAISS_PATH)

                # Generate summary
                summary_prompt = f"Summarize the following text for study notes in 4–5 lines:\n\n{text_content[:1500]}"
                try:
                    summary = llm.invoke(summary_prompt).strip()
                except Exception:
                    summary = "⚠️ Could not generate summary."

                st.session_state.summarized_files.add(uploaded_file.name)

                # Animated summary in main chat
                summary_msg = f"Here’s a quick summary of *{uploaded_file.name}*:\n\n{summary}"
                placeholder = st.empty()
                typed_text = ""
                for char in summary_msg:
                    typed_text += char
                    placeholder.markdown(
                        f"""<div class="flex-container flex-start">
                        <div class='bot-icon'></div>
                        <div class='chat-bubble assistant-bubble'>{typed_text}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)

                st.session_state.messages.append({"role": "assistant", "content": summary_msg})
                memory.chat_memory.add_ai_message(summary_msg)

            else:
                st.warning(f"⚠️ File '{uploaded_file.name}' has no extractable text.")

        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")


# === IMAGES ===
if uploaded_images:
    import numpy as np
    import easyocr
    from transformers import BlipProcessor, BlipForConditionalGeneration

    # Initialize EasyOCR & BLIP
    reader = easyocr.Reader(['en'])
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    for uploaded_image in uploaded_images:
        if uploaded_image.name in st.session_state.summarized_images:
            continue

        img = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(img)  # Convert to numpy array for EasyOCR

        # ---------- OCR Stage ----------
        try:
            ocr_result = reader.readtext(img_np)
            text_content = " ".join([res[1] for res in ocr_result]).strip()
        except Exception:
            text_content = ""

        # ---------- BLIP Caption ----------
        try:
            inputs = processor(images=img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            caption = "⚠️ Could not generate image caption."

        # ---------- Summarize Text if OCR detected ----------
        if text_content:
            summary_prompt = f"Summarize the following text for study notes in 4–5 lines:\n\n{text_content[:1500]}"
            try:
                text_summary = llm.invoke(summary_prompt).strip()
            except Exception:
                text_summary = "⚠️ Could not generate text summary."
            final_msg = f"🖼️ *Image Caption:* {caption}\n\n📝 *Text Summary:* {text_summary}"
        else:
            final_msg = f"🖼️ *Image Caption / Description:* {caption}\n\n📝 No text detected via OCR."

        # ---------- Animated typing effect ----------
        placeholder = st.empty()
        typed_text = ""
        for char in final_msg:
            typed_text += char
            placeholder.markdown(
                f"""<div class="flex-container flex-start">
                <div class='bot-icon'></div>
                <div class='chat-bubble assistant-bubble'>{typed_text}</div>
                </div>""",
                unsafe_allow_html=True
            )
            time.sleep(0.01)

        # ---------- Update Session State ----------
        st.session_state.messages.append({"role":"assistant","content":final_msg})
        memory.chat_memory.add_ai_message(final_msg)
        st.session_state.summarized_images.add(uploaded_image.name)

        # ---------- Success Message under uploader ----------
        images_msg_container.success(f"✅ Image '{uploaded_image.name}' processed successfully!")




# =========================
# MAIN CHAT AREA: Display Messages
# =========================
def display_message(msg):
    if msg["role"]=="user":
        st.markdown(f"""<div class="flex-container flex-end"><div class='chat-bubble user-bubble'>{msg['content']}</div><div class='user-icon'></div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="flex-container flex-start"><div class='bot-icon'></div><div class='chat-bubble assistant-bubble'>{msg['content']}</div></div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    display_message(msg)

# =========================
# MAIN CHAT INPUT AREA
# =========================
user_input = st.chat_input("Ask me anything about your subjects...")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    display_message({"role":"user","content":user_input})

    if user_input.lower() in ["bye","quit","exit"]:
        farewell_msg = "Thank you! Have a great day! 🙂 If you have more questions later, just ask!"
        st.session_state.messages.append({"role":"assistant","content":farewell_msg})
        placeholder = st.empty()
        typed_text = ""
        for char in farewell_msg:
            typed_text += char
            placeholder.markdown(f"""<div class="flex-container flex-start"><div class='bot-icon'></div><div class='chat-bubble assistant-bubble'>{typed_text}</div></div>""", unsafe_allow_html=True)
    else:
        if not is_academic_question(user_input):
            answer = "This question is not related to the study material."
        else:
            memory.chat_memory.messages = []
            for msg in st.session_state.messages:
                if msg["role"]=="user":
                    memory.chat_memory.add_user_message(msg["content"])
                else:
                    memory.chat_memory.add_ai_message(msg["content"])
            result = qa_chain({"question": user_input})
            answer = result["answer"]

        answer = re.sub(r'(https?://[^\s]+)', r'[\1](\1)', answer)

        placeholder = st.empty()
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(f"""<div class="flex-container flex-start"><div class='bot-icon'></div><div class='chat-bubble assistant-bubble'>{typed_text}</div></div>""", unsafe_allow_html=True)
            time.sleep(0.01)

        st.session_state.messages.append({"role":"assistant","content":answer})
        memory.chat_memory.add_ai_message(answer)