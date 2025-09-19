import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from docx import Document

# 1. Setup ChromaDB
client = chromadb.PersistentClient(path="lab_chroma_db")

# 2. Create or get collection
collection = client.get_or_create_collection(
    name="lab_knowledge",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        const API_KEY = process.env.OPENAI_API_KEY;
,
        model_name="text-embedding-3-small"
    )
)

# --- Helpers to load files ---
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def load_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# --- Chunk text into smaller pieces ---
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# --- Main indexing loop ---
docs_folder = "docs"
for filename in os.listdir(docs_folder):
    file_path = os.path.join(docs_folder, filename)
    if filename.endswith(".pdf"):
        text = load_pdf(file_path)
    elif filename.endswith(".docx"):
        text = load_docx(file_path)
    elif filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        continue

    # Split into chunks
    chunks = chunk_text(text)

    # Add each chunk separately
    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_{i}"
        collection.add(
            documents=[chunk],
            ids=[doc_id]
        )

    print(f"✅ Indexed {filename} into {len(chunks)} chunks")

print("🎉 Done! All documents have been indexed in chunks.")
