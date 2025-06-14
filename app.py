# app.py
import faiss
import pickle
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Load FAISS index and texts
index = faiss.read_index("vectorstore/faiss_index")
with open("vectorstore/texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load local LLM (LLaMA)
llm = OllamaLLM(model="llama3")

# Simple prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant for parents asking about programs and courses offered after 10th in India.
    Use the context below to answer the question:

    Context: {context}
    Question: {question}

    Answer:
    """
)

def ask_question(user_question):
    # Embed question using same model
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = embedder.encode([user_question])

    # Search top 3 results
    D, I = index.search(question_embedding, k=3)
    context = "\n\n".join([texts[i] for i in I[0]])

    # Create prompt
    prompt = prompt_template.format(context=context, question=user_question)

    # Get response from local model
    answer = llm.invoke(prompt)
    return answer

# Try it out
if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        response = ask_question(q)
        print(f"\n💬 Answer:\n{response}")

# Ask for feedback
feedback = input("\n👍 Was this answer helpful? (yes/no): ").strip().lower()

# Save feedback to a CSV file (optional)
with open("feedback.csv", "a") as f:
    f.write(f'"{q}","{response.strip()}","{feedback}"\n')

print("✅ Thank you for your feedback!")
