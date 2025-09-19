# utils.py
from pypdf import PdfReader
import docx

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return "\n".join(pages)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text])

def chunk_text(text, chunk_size=1200, overlap=150):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
