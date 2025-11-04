# AAIDC Module-1- RAG
# 🤖 EDUBOT – Intelligent Agentic RAG Assistant

# 📌 Overview  
EDUBOT is an advanced **Retrieval-Augmented Generation (RAG)** and **Agentic AI system** designed for educational and document-based learning.  
It integrates **LangChain**, **LangGraph**, **FAISS**, and **Google Gemini**, enabling seamless document ingestion, continuous monitoring, and interactive Q&A — all inside a beautiful **Streamlit** interface.

The system operates in **two key stages**:  
1️⃣ **Document Ingestion Agent (Backend):** Handles automated loading, embedding, and FAISS vector database updates.  
2️⃣ **RAG + LLM + UI (Frontend):** Provides real-time intelligent question answering with context retrieval, summarization, and memory.


# ✨ Features
| Feature | Description |
| --- | --- |
| 📂 Smart Multi-File Ingestion | Automatically loads and updates TXT, PDF, PPT, DOC, DOCX, XLS, and XLSX files using agentic workflows. |
| 🔁 Auto Vector Update | Continuously monitors the data folder for new or deleted files and updates FAISS vectors dynamically. |
| 🧠 FAISS + MiniLM Embeddings | Uses `all-MiniLM-L6-v2` sentence transformer for efficient context retrieval. |
| 🧩 LangGraph Agent Workflow | Agentic graph automates file detection → ingestion → validation with retries and logging. |
| ⚙️ Gemini-2.0 Flash Integration | Uses Google’s LLM for intelligent, contextual, and educational responses. |
| 🧾 Text + Image Understanding | Extracts text from PDFs, PPTs, DOCs, Excels, and captions images using BLIP + EasyOCR. |
| 🪄 Summarization | Auto-summarizes each uploaded file into concise study notes. |
| 💬 Interactive Chat UI | Beautiful Streamlit interface with animated chat bubbles and color-coded user/assistant messages. |
| 🧮 Evaluation Metrics | Integrated BLEU, ROUGE, and semantic similarity scoring for academic answer evaluation. |
| 📡 Memory-Enabled Conversations | Maintains contextual flow using `ConversationBufferMemory`. |
| 🕵️ Watcher Agent | Continuously monitors the data folder and triggers re-ingestion automatically. |
| ✅ Academic Filter | Restricts to academic queries only; politely blocks unrelated or personal questions. |


# ⚙️ Setup Instructions

# 1️⃣ Install Dependencies
Make sure you have Python 3.11+ installed, then run:

pip install streamlit langchain langgraph faiss-cpu sentence-transformers transformers easyocr google-generativeai evaluate rouge-score python-docx PyPDF2 python-pptx openpyxl pillow python-dotenv


# 2️⃣ Add Documents
Place your TXT, PDF, PPTX, DOCX, or XLSX files inside the Data/ folder.
Ensure PDFs are text-based (not scanned images).


# 3️⃣ Run Document Ingestion Agent
python "Document ingestion.py"


# 4️⃣ Launch the RAG Assistant
streamlit run app.py


# 🖥️ Example Usage

Ask a question:
What are the applications of Artificial Intelligence?

Answer:
Artificial Intelligence (AI) is applied in robotics, healthcare, education, autonomous vehicles, and recommendation systems.  
It enables machines to perform human-like decision-making, perception, and learning.

Sources: ai_notes.pdf

# 🛠️ Tool Integration

## 🔹 Local Tools & Services

### Gemini LLM Integration (Google Gemini)
  - Large language model inference for academic reasoning and summarization
  - Configurable temperature and output tokens for adaptive responses
  - Provides accurate, context-aware, and educational answers

### Document Ingestion Agent (LangGraph + LangChain)
  - Automated multi-format file processing (TXT, PDF, DOCX, PPTX, XLSX)
  - StateGraph-driven workflow: detect → ingest → validate
  - Real-time file watching with auto vector DB updates

### Embedding & Retrieval Engine
  - Embeddings generated via HuggingFace MiniLM (all-MiniLM-L6-v2)
  - Vector indexing and retrieval powered by FAISS
  - Persistent FAISS storage for long-term memory

### Evaluation & Analysis Tools
  - Integrated BLEU, ROUGE, and cosine similarity scoring
  - Automatic semantic similarity tracking for generated responses
  - Logging of ingestion events and evaluation metrics

### Image Understanding Agent
  - OCR extraction using EasyOCR
  - Visual captioning with BLIP (Salesforce/blip-image-captioning-base)
  - Summarization of detected text for study notes

### File Management & Logging
  - Organized FAISS vectorstore with metadata preservation
  - Automated file mapping, update logs, and error handling
  - Continuous monitoring by Watcher Agent for changes in Data folder

# 💻 System Requirements

• Operating System: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+  
• Python Version: 3.10 or higher (tested on 3.11)  
• RAM: Minimum 8 GB (16 GB recommended for faster embedding and LLM inference)  
• Storage: 5–10 GB free (for vector DB, logs, and local documents)  
• GPU (Optional): NVIDIA GPU with CUDA support for BLIP and EasyOCR acceleration  
• Dependencies: Refer to requirements.txt or setup instructions above  

# ⚙️ Tech Stack

• LLM: Google Gemini 2.0 Flash  
• Frameworks: LangChain, LangGraph, Streamlit  
• Embeddings: HuggingFace MiniLM (all-MiniLM-L6-v2)  
• Vector Database: FAISS (local persistent store)  
• OCR & Image Captioning: EasyOCR, BLIP (Salesforce)  
• Document Loaders: LangChain Unstructured, PyPDFLoader, Docx2txt, PowerPoint, Excel loaders  
• Evaluation Metrics: BLEU, ROUGE, Cosine Similarity  
• Memory: ConversationBufferMemory (LangChain)  
• Logging: Auto timestamped logs for ingestion & updates  
• UI: Streamlit with custom HTML/CSS chat interface  


# 📊 Highlights

✅ Agentic document ingestion using LangGraph workflow (detect → ingest → validate)  
✅ Real-time RAG assistant powered by Google Gemini 2.0 Flash  
✅ Multi-file support with auto text extraction (PDF, DOCX, PPTX, XLSX, TXT)  
✅ Memory-based conversation management for contextual responses  
✅ Semantic evaluation using BLEU, ROUGE, and cosine similarity metrics  
✅ Integrated image-to-text and captioning (EasyOCR + BLIP)  
✅ Auto logging of ingestion activity and FAISS vector updates  
✅ Modern Streamlit UI with chat history, new chat, and logout features  


# 🧾 Performance & Metrics

⚡ Avg. Response Time: 2–4 seconds (text)  
📊 Semantic Similarity: ≥ 0.85 (average on reference-based tests)  
🧮 Evaluation Metrics: BLEU, ROUGE-L, and Cosine Similarity  
🧠 Memory Retention: Full conversation buffer (preserves context during chat)  


# 🪪 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
You are free to use, modify, and distribute this software under the same license terms.  


# 🙌 Acknowledgements

🔹 **LangChain / LangGraph** — For building the ingestion and retrieval orchestration backbone.  
🔹 **Hugging Face** — For providing open-source embedding and summarization models.  
🔹 **Google Gemini** — For powering the LLM responses with contextual reasoning.  
🔹 **Streamlit** — For creating an elegant and interactive user interface.  
🔹 **AAIDC Module 2 Program** — For project structure, certification guidance, and evaluation standards.  
