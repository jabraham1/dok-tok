import chromadb
from chromadb.utils import embedding_functions
import os

# --- Connect to ChromaDB ---
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="lab_docs",
embedding_function=embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

    
    )

# --- Example query ---
query_text = "What does high Magnesium mean"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

print("\n🔎 Query:", query_text)
for doc, score in zip(results['documents'][0], results['distances'][0]):
    print(f"\nMatch (score={score:.4f}):\n{doc}")
