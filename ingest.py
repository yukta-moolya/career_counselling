# ingest.py
import os
import faiss
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_PATH = "data\\Career_Handbook.pdf"
VECTOR_PATH = "vectorstore\\faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"


# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(VECTOR_PATH), exist_ok=True)


# Step 1: Load PDF
reader = PdfReader(DATA_PATH)
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text()

# Step 2: Chunk text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_text(raw_text)

# Step 3: Embed using Sentence Transformers
embedder = SentenceTransformer(MODEL_NAME)
embeddings = embedder.encode(texts)

# Step 4: Create FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# Save index and texts
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, VECTOR_PATH)
with open("vectorstore/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Ingestion complete. FAISS index created.")
