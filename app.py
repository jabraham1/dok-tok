import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pypdf import PdfReader
from docx import Document
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

templates = Jinja2Templates(directory="templates")

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Save file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    if file.filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        return HTMLResponse("<h3>❌ Unsupported file format. Please upload PDF or DOCX.</h3>")

    # AI interpretation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a functional medicine doctor explaining lab results in simple, supportive language. Avoid jargon, and give practical guidance."},
            {"role": "user", "content": f"Interpret the following lab report for the patient:\n\n{text}"}
        ]
    )
    ai_answer = response.choices[0].message.content

    # Clean up
    os.remove(file_path)

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "interpretation": ai_answer}
    )

