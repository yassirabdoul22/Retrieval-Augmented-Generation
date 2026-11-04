from typing import List, Optional
import numpy as np
from src.models import Chunk
from src.utils.validator import Validator
from src.retrieval.base import Retriever
from src.utils.config import (
    EMBEDDING_MODEL,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_FILE,
    CHUNKS_FILE
)
from src.utils.exceptions import IndexPersistenceError, RetrievalError
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle


class EmbeddingRetriever(Retriever):
    """embedding Retriever"""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vect_matrix: Optional[np.ndarray] = None
        self._model: Optional[SentenceTransformer] = None
        self._validator = Validator()

    @property
    def chunks(self) -> List[Chunk]:
        """Return the indexed chunks."""
        return self._chunks

    def index(self, chunks: List[Chunk]) -> None:
        model = self._load_model()
        self._chunks = chunks
        chunks_content = [chunk.indexable_text for chunk in chunks]
        self._vect_matrix = model.encode(
            chunks_content,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        k = self._validator.validate_top_k_value(k, len(self._chunks))
        query = self._validator.validate_prompt(query)
        model = self._load_model()
        query_vect = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        scores = self._vect_matrix @ query_vect
        top = np.argsort(-scores)[:k]
        return [self._chunks[i] for i in top]

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                self._model.max_seq_length = EMBEDDING_MAX_TOKENS
            except (OSError, ValueError) as e:
                raise RetrievalError(
                    f"cannot download or load '{EMBEDDING_MODEL}', "
                    "check your internet connection and retry"
                ) from e
        return self._model

    def save(self, path: str) -> None:
        if self._vect_matrix is None:
            raise IndexPersistenceError("nothing to save, run index first")
        try:
            np.save(Path(path) / EMBEDDING_FILE, self._vect_matrix)
        except OSError:
            raise IndexPersistenceError("embedding index result not saved")

    def load(self, path: str) -> None:
        try:
            directory = Path(path)

            self._vect_matrix = np.load(directory / EMBEDDING_FILE)

            with open(directory / CHUNKS_FILE, "rb") as file:
                self._chunks = pickle.load(file)

            if len(self._vect_matrix) != len(self._chunks):
                raise IndexPersistenceError(
                    "embeddings does not match chunks "
                    "try to run index --hybrid again")

        except FileNotFoundError:
            raise IndexPersistenceError(
                f"no embedding in {path}, run index --hybrid first")
        except (IsADirectoryError, NotADirectoryError):
            raise IndexPersistenceError(
                f"invalid path {path} for emdeddings files "
                "try to use a correct path")
        except PermissionError:
            raise IndexPersistenceError(
                f"cannot read file {path} (u don't have permissions)")
        except (ValueError, EOFError, pickle.UnpicklingError):
            raise IndexPersistenceError(
                f"corrupted index in '{path}', run index --hybrid again")
        except OSError as e:
            raise IndexPersistenceError(f"{e}")
