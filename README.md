# DocMind RAG Assistant

A full-stack, domain-specific Retrieval-Augmented Generation (RAG) assistant that answers questions strictly from uploaded PDF documents with exact page-level citations.

## Key Features
- **In-App PDF Upload:** Upload any text-based PDF from the Streamlit sidebar and index it into ChromaDB via FastAPI.
- **Deduplicated Vector Ingestion:** Chunks PDFs and stores embeddings locally in ChromaDB using MD5 deduplication.
- **Grounded Answers with Citations:** Uses Groq (`openai/gpt-oss-20b`) to answer strictly from retrieved context and returns source snippets with page numbers.

## Tech Stack
- **Backend:** FastAPI, Uvicorn, Pydantic
- **RAG & Vector Store:** LangChain, ChromaDB, `all-MiniLM-L6-v2` embeddings
- **LLM:** Groq API (OpenAI-compatible SDK)
- **Frontend:** Streamlit

## Quick Start
1. Clone the repo and install dependencies:
   - `git clone https://github.com/student-pranav/docmind-rag.git`
   - `cd docmind-rag`
   - `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
3. Start the FastAPI backend (Terminal 1): `uvicorn app.main:app --reload`
4. Start the Streamlit UI (Terminal 2): `streamlit run ui.py`
5. Open `http://localhost:8501`, upload a PDF in the sidebar, click **Upload & Ingest PDF**, and start chatting.

## Planned Roadmap
- Hybrid search (BM25 + dense vector retrieval)
- Automated RAG evaluation suite (`evals/`) for retrieval hit rate and faithfulness
- Multi-document management and Dockerization
