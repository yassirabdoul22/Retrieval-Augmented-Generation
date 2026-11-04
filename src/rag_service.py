"""Single-query RAG: retrieve sources, then generate a grounded answer."""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.generator import Generator

from src.models import (
    MinimalSearchResults, MinimalAnswer, MinimalSource, UnansweredQuestion,
)
from src.retrieval.base import Retriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.retriever_cache import RetrieverCache
from src.dataset_answerer import DatasetAnswerer
from src.utils.config import DEFAULT_INDEX_PATH


def build_retriever(hybrid: bool, use_cache: bool) -> Retriever:
    """Return BM25, or BM25 + embeddings fused by RRF, maybe cached."""
    retriever: Retriever = BM25Retriever()
    if hybrid:
        from src.retrieval.embedding_retriever import EmbeddingRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(retriever, EmbeddingRetriever())
    if use_cache:
        retriever = RetrieverCache(retriever)
    return retriever


class RagService:
    """Answers a question; the model and index are loaded on first use."""

    def __init__(
        self, index_path: str = DEFAULT_INDEX_PATH, use_cache: bool = False,
        hybrid: bool = False,
    ) -> None:
        """Remember the index location without loading anything yet."""
        self._index_path = index_path
        self._use_cache = use_cache
        self._hybrid = hybrid
        self._retriever: Optional[Retriever] = None
        self._generator: Optional[Generator] = None

    def search(self, query: str, k: int) -> MinimalSearchResults:
        """Retrieve source locations without loading generation models."""
        question = UnansweredQuestion(question=query)
        if self._retriever is None:
            retriever = build_retriever(self._hybrid, self._use_cache)
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
        if self._generator is None:
            from src.generator import Generator

            self._generator = Generator()
        return DatasetAnswerer(self._generator).answer_question(result)

    def answer(self, query: str, k: int) -> str:
        """Retrieve sources and return the generated answer text."""
        return self.answer_result(query, k).answer
