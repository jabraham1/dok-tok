import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os

# --- Connect to OpenAI ---
client_gpt = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Connect to Chroma ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="lab_docs",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
)

# --- Query function ---
def ask_gpt(question: str):
    # Step 1. Search docs
    results = collection.query(query_texts=[question], n_results=3)
    docs = results['documents'][0]

    # Step 2. Build context
    context = "\n\n".join(docs)

    # Step 3. Send to GPT
    response = client_gpt.chat.completions.create(
        model="gpt-4o-mini",  # cost-efficient + smart
        messages=[
            {"role": "system", "content": "You are a functional medicine AI. Interpret labs with emphasis on root cause, nutrition, and lifestyle. Exlain labs in a clear, patient-friendly way."},

            {"role": "user", "content": f"Question: {question}\n\nRelevant Lab Notes:\n{context}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# --- Example run ---
if __name__ == "__main__":
    q = "Explain the meaning of elevated anti-TPO in simple terms."
    print("💡", ask_gpt(q))
