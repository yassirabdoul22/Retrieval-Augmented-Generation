from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.utils.config import DEFAULT_TOP_K
from src.rag_service import RagService

app = FastAPI()
rag_service = RagService()


class Request(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=DEFAULT_TOP_K, ge=1)


class RequestAnswer(BaseModel):
    answer: str


@app.post("/answer", response_model=RequestAnswer)
def answer(request: Request) -> RequestAnswer:
    result = rag_service.answer(
        query=request.query,
        k=request.k,
    )
    if isinstance(result, str):
        return RequestAnswer(answer=result)

    