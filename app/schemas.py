from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question about the documents")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of source chunks to retrieve")


class Citation(BaseModel):
    source: str
    page: int
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
