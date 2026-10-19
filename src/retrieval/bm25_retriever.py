import pickle
from pathlib import Path
from typing import List

import bm25s

from src.utils.exceptions import IndexPersistenceError
from src.models import Chunk
from .base import Retriever
from src.utils.validator import Validator

CHUNKS_FILE = "chunks.pkl"


class BM25Retriever(Retriever):

    """Rank and persist source chunks using lexical BM25 scoring."""

    def __init__(self) -> None:
        """Create an empty BM25 index and retrieval input validator."""
        self.chunks: List[Chunk] = []
        self.bm25s_index = bm25s.BM25()
        self._validator = Validator()

    def index(self, chunks: List[Chunk]) -> None:
        """Tokenize source chunks and build their BM25 scoring index."""
        content: List[str] = [chunk.indexable_text for chunk in chunks]
        corpus_tokens = bm25s.tokenize(content)
        self.bm25s_index.index(corpus_tokens)
        self.chunks = chunks

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        """Return the top ranked matching chunks for a query."""
        k = self._validator.validate_top_k_value(k, len(self.chunks))
        query_tokens = bm25s.tokenize(query)
        indexes, scores = self.bm25s_index.retrieve(query_tokens, k=k)
        return [
            self.chunks[idx]
            for idx, score in zip(indexes[0], scores[0])
            if score > 0
        ]

    def save(self, path: str) -> None:
        """Persist the component state at the requested path."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.bm25s_index.save(path)
            with open(Path(path) / CHUNKS_FILE, "wb") as file:
                pickle.dump(self.chunks, file)
        except OSError as e:
            raise IndexPersistenceError(f"{e}") from e

    def load(self, path: str) -> None:
        """Restore the component state from the requested path."""
        try:
            self.bm25s_index = bm25s.BM25.load(path)
            with open(Path(path) / CHUNKS_FILE, "rb") as file:
                self.chunks = pickle.load(file)
        except FileNotFoundError:
            raise IndexPersistenceError(
                f"no index in '{path}', run index first"
            )
        except PermissionError:
            raise IndexPersistenceError(
                f"you don't have permission to read '{path}'"
            )
        except (IsADirectoryError, NotADirectoryError):
            raise IndexPersistenceError(f"invalid index path '{path}'")
        except OSError as e:
            raise IndexPersistenceError(f"{e}")
