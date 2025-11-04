# -*- coding: utf-8 -*-
"""
SMART EDUCATIONAL CHATBOT (EDUBOT)
With Fixed Memory + Strict Q-Type Control + Better Context Fallback + Semantic Similarity Only
Created: 2025-08-29
Updated: 2025-10-24
@author: Arun
"""

# ============================ #
# Step 0: Imports
# ============================ #
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import csv
import re
import pickle

# Evaluation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ============================ #
# Step 1: Paths
# ============================ #
DATA_PATH = r"D:\AAIDC\Project 1\Data"
DB_FAISS_PATH = r"D:\AAIDC\Project 1\vectorstore"
FILE_MAPPING_PATH = r"D:\AAIDC\Project 1\file_mapping.pkl"
LOG_FILE = r"D:\AAIDC\Project 1\update_log.txt"

# ============================ #
# Step 2: Google API Key
# ============================ #
from dotenv import load_dotenv
import os

# Load .env file using absolute path
load_dotenv(r"D:\AAIDC\Project 1\.env")  

# Access API key
API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================ #
# Step 3: Load FAISS DB + Embeddings
# ============================ #
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})

# ============================ #
# Step 4: Gemini LLM
# ============================ #
llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=API_KEY,
    temperature=0.2,
    max_output_tokens=1200
)

# ============================ #
# Step 5: Prompt Template
# ============================ #
FULL_PROMPT_TEMPLATE = """
You are EDUBOT, an AI tutor for K-12 students. 
Act like a flexible teacher who adapts explanations to the student’s intent.

---

### Memory & Context Rules:
- Always use **chat history** to interpret vague follow-ups (e.g., "it", "this", "go with that").
- Continue the flow instead of repeating the same explanation.
- If FAISS context is weak, **fallback to general academic knowledge**.
- If the student gives acknowledgments like "okay", "yes", "continue", interpret them as **follow-up requests**.

---

### Question-Type Rules (strict):
- If the question starts with **Who** → answer only *who (person, group, entity)* with background, role, contributions, legacy.
- If the question starts with **What** → answer only *what (definition, fact, meaning)* with scope, uses, and applications.
- If the question starts with **When** → answer only *time-related details*.
- If the question starts with **Why** → answer only *reasons/importance*.
- If the question starts with **How** → answer only *steps, process, or explanation*.
- Do not mix categories unless the student explicitly asks.

---

### Depth Control:
- Always expand answers into **at least 4–5 lines**.
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

FULL_PROMPT = PromptTemplate(
    template=FULL_PROMPT_TEMPLATE,
    input_variables=["chat_history", "context", "question"]
)

# ============================ #
# Step 6: Memory
# ============================ #
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ============================ #
# Step 7: Conversational Retrieval Chain
# ============================ #
pdf_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": FULL_PROMPT},
    return_source_documents=False,
    output_key="answer"
)

# ============================ #
# Step 8: Semantic Similarity Function
# ============================ #
def semantic_similarity_score(reference, generated, embed_model=embeddings):
    """Compute cosine similarity between reference and generated answer."""
    if not reference.strip() or not generated.strip():
        return None
    ref_vec = embed_model.embed_query(reference)
    gen_vec = embed_model.embed_query(generated)
    score = cosine_similarity([ref_vec], [gen_vec])[0][0]
    return round(score, 4)

# ============================ #
# Step 9: Academic Question Heuristic
# ============================ #
def is_academic_question(question):
    """Detect if question is academic or follow-up."""
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
# ============================ #
# Step 10: Chat Loop with User-Controlled Debug
# ============================ #
print("✅ EDUBOT ready! Type 'exit' or 'quit' to stop.")

# Ask user once whether to enable debug and history
debug_input = input("Do you want to see retrieved context? (yes/no): ").strip().lower()
DEBUG_MODE = True if debug_input in ["yes", "y"] else False

history_input = input("Do you want to see full chat history? (yes/no): ").strip().lower()
DEBUG_HISTORY = True if history_input in ["yes", "y"] else False

with open(LOG_FILE, "a", encoding="utf-8") as log_f, \
     open("edubot_logs.csv", "a", newline="", encoding="utf-8") as csv_f:

    writer = csv.writer(csv_f)
    writer.writerow(["Query", "Answer", "SemanticSim"])

    while True:
        query = input("\n🟢 Your question: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye! Have a great day!")
            break

        # Academic check
        if not is_academic_question(query):
            answer = "This question is not related to the study material."
            print("\n🟢 Question:\n", query)
            print("\n💬 Answer:\n", answer)
            writer.writerow([query, answer, None])
            log_f.write(f"Query: {query}\nAnswer: {answer}\nSemanticSim: None\n\n")
            continue

        # Retrieve context
        docs = retriever.get_relevant_documents(query)
        context_text = " ".join([doc.page_content for doc in docs]) if docs else ""
        if not context_text.strip():
            context_text = "General academic knowledge (fallback)."

        # Run through chain
        try:
            result = pdf_chain({"question": query})
            answer = result["answer"].strip()
        except Exception as e:
            print("❌ Error generating answer:", e)
            continue

        # Print output
        print("\n🟢 Question:\n", query)
        if DEBUG_MODE:
            print("\n🔎 [DEBUG] Retrieved Context (first 300 chars):")
            print(context_text[:300] + "..." if context_text else "⚠️ No context retrieved")

        print("\n💬 Answer:\n", answer)

        if DEBUG_HISTORY:
            print("\n📚 [DEBUG] Full Chat History:")
            full_history = memory.load_memory_variables({}).get("chat_history", [])
            for i, msg in enumerate(full_history, 1):
                print(f"{i}. {msg.type}: {msg.content}")

        # Semantic similarity evaluation
        similarity = semantic_similarity_score(context_text, answer)
        print(f"\n📊 Semantic Similarity: {similarity}")

        # Log
        writer.writerow([query, answer, similarity])
        log_f.write(f"Query: {query}\nAnswer: {answer}\nSemanticSim: {similarity}\n\n")
