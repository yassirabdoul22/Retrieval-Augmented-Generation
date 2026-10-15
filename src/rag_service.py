"""Single-query RAG: retrieve sources, then generate a grounded answer."""

from typing import Optional

from src.generator import Generator
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

    def answer(self, query: str, k: int) -> str:
        """Retrieve the top-k sources for query and answer from them."""
        if self._retriever is None:
            retriever = BM25Retriever()
            retriever.load(self._index_path)
            self._retriever = retriever
        sources = self._retriever.retrieve(query=query, k=k)
        source_text = [read_source_text(source) for source in sources]
        if self._generator is None:
            self._generator = Generator()
        return self._generator.generate(query=query, source_text=source_text)
