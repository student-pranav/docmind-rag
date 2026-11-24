import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "DocMind RAG Assistant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
PDF_PATH = os.getenv("PDF_PATH", "./data/doc.pdf")
