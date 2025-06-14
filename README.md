# Career Counselling Assistant (RAG-based QA with Ollama)

This project is an AI assistant for answering parent Doubt & Questions about school programs, partnerships, and pricing. It uses a PDF document as knowledge and a local LLaMA model for answers.

## 🔧 Features

- Loads program details from a PDF (Manitoba sample)
- Builds a FAISS vector store using `sentence-transformers`
- Answers questions using a locally run LLaMA model (via Ollama)
- Collects feedback after each answer

## 🛠️ Technologies Used

- Python
- LangChain
- FAISS
- SentenceTransformers
- LLaMA 3 (via Ollama)
- PyPDF2

## 📁 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt

run python ingest.py
and
run python app.py

Ask your questions and provide feedback!