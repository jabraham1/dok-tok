from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# ✅ Enable CORS so frontend fetch calls work
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can later restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts an uploaded lab file and returns a placeholder interpretation.
    Later, you can plug in OpenAI or your analysis logic here.
    """
    contents = await file.read()
    # For now, just return file metadata
    return {
        "filename": file.filename,
        "size_bytes": len(contents),
        "message": "File received successfully. Analysis goes here."
    }

# ✅ Optional local run for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

