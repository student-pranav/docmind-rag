import hashlib
from openai import OpenAI
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.config import GROQ_API_KEY, GROQ_MODEL, CHROMA_DIR, PDF_PATH

_client = None
_vector_store = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        _vector_store = Chroma(
            collection_name="docmind_docs",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vector_store


def ingest_pdf(pdf_path: str = PDF_PATH) -> int:
    reader = PdfReader(pdf_path)
    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": pdf_path, "page": i + 1},
        )
        for i, page in enumerate(reader.pages)
    ]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=150,
        chunk_overlap=20,
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]

    ids = [
        hashlib.md5(
            f"{c.metadata.get('source')}:{c.metadata.get('page')}:{c.page_content}".encode()
        ).hexdigest()
        for c in chunks
    ]

    store = get_vector_store()
    if chunks:
        store.add_documents(chunks, ids=ids)
    return len(chunks)


def answer_question(question: str, top_k: int = 3) -> tuple[str, list[dict]]:
    store = get_vector_store()
    results = store.similarity_search_with_score(question, k=top_k)

    citations = []
    context_blocks = []
    for idx, (doc, _score) in enumerate(results, start=1):
        source = str(doc.metadata.get("source", PDF_PATH))
        page = int(doc.metadata.get("page", 1))
        snippet = doc.page_content.strip()[:220]
        citations.append({"source": source, "page": page, "snippet": snippet})
        context_blocks.append(f"[Source {idx} | {source}, Page {page}]\n{doc.page_content.strip()}")

    context_text = "\n\n---\n\n".join(context_blocks)
    system_prompt = (
        "You are a helpful domain assistant. Answer ONLY using the provided sources. "
        "If the answer is not in the sources, say you don't have enough information in the document. "
        "Mention the source/page when referencing facts."
    )

    resp = get_client().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Sources:\n{context_text}\n\nQuestion: {question}"},
        ],
    )
    answer = resp.choices[0].message.content or ""
    return answer, citations
