# Career Counselling Assistant (RAG-based QA with Ollama)

This project is an AI-powered assistant designed to answer parents' and students' questions about career options after 10th grade in India. It uses a PDF document as its knowledge base and leverages a locally hosted LLaMA model (via Ollama) to provide accurate, contextual responses using Retrieval-Augmented Generation (RAG). The system is built using LangChain, FAISS, and includes a feedback mechanism to log user satisfaction.

## 🔧 Features

- Loads program details from a PDF
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
