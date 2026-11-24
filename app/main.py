from fastapi import FastAPI, HTTPException
from app.config import APP_NAME, GROQ_API_KEY
from app.schemas import ChatRequest, ChatResponse, Citation
from app.rag.pipeline import ingest_pdf, answer_question

app = FastAPI(title=APP_NAME, version="0.2.0")


@app.get("/health")
async def health_check():
    return {"status": "ok", "app": APP_NAME, "groq_key_set": bool(GROQ_API_KEY)}


@app.post("/ingest")
async def ingest():
    try:
        chunks_count = ingest_pdf()
        return {"status": "ingested", "chunks_processed": chunks_count}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF file not found at ./data/doc.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is missing in .env")
    try:
        answer, raw_citations = answer_question(request.question, top_k=request.top_k)
        return ChatResponse(
            answer=answer,
            citations=[Citation(**c) for c in raw_citations],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
