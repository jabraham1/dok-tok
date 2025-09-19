import os
import base64
import fitz  # PyMuPDF for PDF text extraction
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

# Load API key safely from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend (place index.html inside "static/")
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts a PDF or image upload, extracts lab data, and returns
    a functional medicine interpretation from OpenAI.
    """
    try:
        if file.content_type == "application/pdf":
            # Extract text from PDF
            pdf_bytes = await file.read()
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in pdf_doc)

            user_prompt = (
                "You are a functional medicine clinician. "
                "Explain these lab results in a clear, patient-friendly way. "
                "Interpret with emphasis on root cause, nutrition, and lifestyle. "
                "Include likely causes, when to seek follow-up, and simple lifestyle or supplement suggestions.\n\n"
                f"Lab report text:\n{text}"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=1000
            )
            return {"result": response.choices[0].message.content}

        elif file.content_type.startswith("image/"):
            # Convert image to base64 for GPT-4o Vision
            img_bytes = await file.read()
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            img_data_url = f"data:{file.content_type};base64,{b64_img}"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text":
                            "You are a functional medicine clinician. "
                            "Explain these lab results in a kind, patient-friendly way. "
                            "Focus on root causes, nutrition, and lifestyle. "
                            "Provide likely explanations, follow-up needs, and simple practical advice."
                        },
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

