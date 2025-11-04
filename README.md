# AAIDC Module-1- RAG
# 🤖 EDUBOT – Intelligent Agentic RAG Assistant

📌 Overview  
EDUBOT is an advanced **Retrieval-Augmented Generation (RAG)** and **Agentic AI system** designed for educational and document-based learning.  
It integrates **LangChain**, **LangGraph**, **FAISS**, and **Google Gemini**, enabling seamless document ingestion, continuous monitoring, and interactive Q&A — all inside a beautiful **Streamlit** interface.

The system operates in **two key stages**:  
1️⃣ **Document Ingestion Agent (Backend):** Handles automated loading, embedding, and FAISS vector database updates.  
2️⃣ **RAG + LLM + UI (Frontend):** Provides real-time intelligent question answering with context retrieval, summarization, and memory.


✨ Features
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


📂 Project Structure
