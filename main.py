import os
import base64
import fitz  # PyMuPDF for PDF text extraction
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if file.content_type == "application/pdf":
            # Extract text from PDF
            pdf_bytes = await file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in pdf_doc)
            user_prompt = f"Please analyze the following lab report text:\n{text}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000
            )
            return {"result": response.choices[0].message.content}

        elif file.content_type.startswith("image/"):
            # Convert image to base64
            img_bytes = await file.read()
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            img_data_url = f"data:{file.content_type};base64,{b64_img}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "You are a functional medicine clinician. 
Explain the actual results of their labs in a kind, patient-friendly way. 
Interpret labs with emphasis on root cause, nutrition, and lifestyle. 
Include likely causes, when to seek follow-up, and simple lifestyle or supplement suggestions."},
                        {"type": "image_url", "image_url": {"url": img_data_url}}
                    ]}
                ],
                max_tokens=1000
            )
            return {"result": response.choices[0].message.content}

        else:
            return {"error": "Unsupported file type. Please upload a PDF or an image."}

    except Exception as e:
        return {"error": str(e)}
