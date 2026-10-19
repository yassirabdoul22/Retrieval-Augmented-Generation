"""Local HTTP access to retrieval and grounded answers."""

from fastapi import FastAPI, Request as HttpRequest
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models import MinimalSearchResults
from src.utils.config import DEFAULT_TOP_K
from src.utils.exceptions import (
    RetrievalError, IndexingError, IndexPersistenceError,
)
from src.rag_service import RagService

app = FastAPI()
rag_service = RagService()


class Request(BaseModel):
    """A question and the requested number of retrieved sources."""

    query: str = Field(min_length=1)
    k: int = Field(default=DEFAULT_TOP_K, ge=1)


class RequestAnswer(BaseModel):
    """Generated answer returned by the answer endpoint."""

    answer: str


@app.exception_handler(RetrievalError)
@app.exception_handler(IndexingError)
@app.exception_handler(ValueError)
def handle_error(request: HttpRequest, exc: Exception) -> JSONResponse:
    """Translate expected pipeline errors into HTTP error responses."""
    status = 503 if isinstance(exc, IndexPersistenceError) else 400
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.post("/search", response_model=MinimalSearchResults)
def search(request: Request) -> MinimalSearchResults:
    """Return ranked source locations for one question."""
    return rag_service.search(query=request.query, k=request.k)


@app.post("/answer", response_model=RequestAnswer)
def answer(request: Request) -> RequestAnswer:
    """Generate a grounded answer using the shared RAG service."""
    return RequestAnswer(answer=rag_service.answer(
        query=request.query, k=request.k,
    ))
