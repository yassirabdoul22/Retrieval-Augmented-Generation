"""Single-query RAG: retrieve sources, then generate a grounded answer."""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.generator import Generator

from src.models import (
    MinimalSearchResults, MinimalAnswer, MinimalSource, UnansweredQuestion,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.config import DEFAULT_INDEX_PATH
from src.utils.io_utils import read_source_text


class RagService:
    """Answers a question; the model and index are loaded on first use."""

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH) -> None:
        """Remember the index location without loading anything yet."""
        self._index_path = index_path
        self._retriever: Optional[BM25Retriever] = None
        self._generator: Optional[Generator] = None

    def search(self, query: str, k: int) -> MinimalSearchResults:
        """Retrieve source locations without loading generation models."""
        question = UnansweredQuestion(question=query)
        if self._retriever is None:
            retriever = BM25Retriever()
            retriever.load(self._index_path)
            self._retriever = retriever
        sources = self._retriever.retrieve(query=query, k=k)
        return MinimalSearchResults(
            question_id=question.question_id, question=query,
            retrieved_sources=[
                MinimalSource.model_validate(source) for source in sources],
        )

    def answer_result(self, query: str, k: int) -> MinimalAnswer:
        """Return an answer together with the retrieved source locations."""
        result = self.search(query, k)
        source_text = [read_source_text(s) for s in result.retrieved_sources]
        if self._generator is None:
            from src.generator import Generator

            self._generator = Generator()
        answer = self._generator.generate(query=query, source_text=source_text)
        return MinimalAnswer(**result.model_dump(), answer=answer)

    def answer(self, query: str, k: int) -> str:
        """Retrieve sources and return the generated answer text."""
        return self.answer_result(query, k).answer
