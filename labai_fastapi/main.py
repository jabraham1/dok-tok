# main.py
import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from chromadb.utils import embedding_functions
import chromadb
from openai import OpenAI
from utils import extract_text_from_pdf, extract_text_from_docx, chunk_text

# Config via env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "/data/chroma_db")  # must be persistent volume
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/data/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# Init OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Init Chroma persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="labs_collection",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
)

app = FastAPI(title="LabAI - FastAPI")

def index_file_chunks(filename, text):
    """Chunk and add to Chroma (safe to call in background)."""
    chunks = chunk_text(text)
    ids = [f"{filename}__{i}" for i in range(len(chunks))]
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        slice_docs = chunks[i:i+batch_size]
        slice_ids = ids[i:i+batch_size]
        collection.add(documents=slice_docs, ids=slice_ids, metadatas=[{"source": filename}] * len(slice_docs))

@app.post("/upload")
async def upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Validate extension
    fname = file.filename
    if not fname.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    # Save to disk
    token = uuid.uuid4().hex
    saved_name = f"{token}_{fname}"
    saved_path = os.path.join(UPLOAD_FOLDER, saved_name)
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text (do a quick check to ensure extractable text)
    if fname.lower().endswith(".pdf"):
        text = extract_text_from_pdf(saved_path)
    elif fname.lower().endswith(".docx"):
        text = extract_text_from_docx(saved_path)
    else:
        # txt
        with open(saved_path, "r", encoding="utf-8") as fh:
            text = fh.read()

    if not text or len(text.strip()) == 0:
        return JSONResponse({"ok": False, "error": "No extractable text"}, status_code=400)

    # Background index (so upload returns quickly)
    background_tasks.add_task(index_file_chunks, saved_name, text)

    return {"ok": True, "file": saved_name, "message": "File uploaded and will be indexed shortly."}

@app.post("/interpret")
async def interpret(filename: str):
    """Query stored chunks related to this filename and ask LLM to interpret them."""
    # Retrieve recent chunks for filename
    query = f"Interpret the lab data from file {filename}. Provide a patient-friendly summary, likely causes, and recommended next steps."
    results = collection.query(query_texts=[query], n_results=4)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return {"ok": False, "error": "No context found for that file yet."}
    context = "\n\n".join(docs[:6])  # limit

    # Ask OpenAI (new client)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a functional medicine clinician. Explain labs in simple patient-friendly language. Include likely causes and practical suggestions."},
            {"role": "user", "content": f"Context:\n{context}\n\nNow summarize for the patient."}
        ],
        temperature=0.2,
        max_tokens=800
    )

    ai_text = response.choices[0].message.content
    return {"ok": True, "interpretation": ai_text}
