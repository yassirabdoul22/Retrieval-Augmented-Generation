import pickle
from typing import List

import bm25s

from src.exceptions import IndexPersistenceError
from src.models import Chunk

from .base import Retriever


class BM25Retriever(Retriever):

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.bm25s_index = bm25s.BM25()

    def index(self, chunks: List[Chunk]) -> None:
        content: List[str] = [chunk.text for chunk in chunks]
        corpus_tokens = bm25s.tokenize(content)
        self.bm25s_index.index(corpus_tokens)
        self.chunks = chunks

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        if k <= 0:
            return []
        query_tokens = bm25s.tokenize(query)
        indexes, _ = self.bm25s_index.retrieve(query_tokens, k=k)
        return [self.chunks[idx] for idx in indexes[0]]

    def save(self, path: str) -> None:
        try:
            self.bm25s_index.save(path)
            with open(f"{path}/chunks.pkl", "wb") as file:
                pickle.dump(self.chunks, file)
        except OSError as e:
            raise IndexPersistenceError(f"{e}") from e

    def load(self, path: str) -> None:
        try:
            self.bm25s_index = bm25s.BM25.load(path)
            with open(f"{path}/chunks.pkl", "rb") as file:
                self.chunks = pickle.load(file)
        except OSError as e:
            raise IndexPersistenceError(f"{e}") from e
