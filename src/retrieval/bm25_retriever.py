import pickle
from typing import List

import bm25s

from src.exceptions import IndexPersistenceError, InvalidTopKError
from src.models import Chunk
from .base import Retriever
from src.validator import Validator

class BM25Retriever(Retriever):

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.bm25s_index = bm25s.BM25()
        self._validator = Validator()
    def index(self, chunks: List[Chunk]) -> None:
        content: List[str] = [chunk.indexable_text for chunk in chunks]
        corpus_tokens = bm25s.tokenize(content)
        self.bm25s_index.index(corpus_tokens)
        self.chunks = chunks
    
    def retrieve(self, query: str, k: int, validator: Validator) -> List[Chunk]:
        
        k= validator.validate_top_k_value(k,len(self.chunks))
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
        except FileNotFoundError:
            raise IndexPersistenceError(f"no index in '{path}' , run index first")
        except PermissionError:
            raise IndexPersistenceError(f"you didn't have permission to read '{path}'")
        except (IsADirectoryError, NotADirectoryError):
            raise IndexPersistenceError(f"invalid index path '{path}'")
        except OSError as e:
            raise IndexPersistenceError(f"{e}")

