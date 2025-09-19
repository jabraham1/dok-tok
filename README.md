LabAI - FastAPI lab interpreter
Start (local): uvicorn main:app --reload
Endpoints:
 - POST /upload (multipart form: file=PDF or DOCX) -> returns interpretation page
Notes: set OPENAI_API_KEY in environment or .env
