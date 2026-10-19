from abc import ABC, abstractmethod
from typing import List

from src.models import Chunk


class Retriever(ABC):

    """Define the interface for indexing and retrieving source chunks."""
    chunks: List[Chunk]

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None:
        """Build a searchable index from the supplied source chunks."""
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int) -> List[Chunk]:
        """Return the top ranked matching chunks for a query."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the component state at the requested path."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the component state from the requested path."""
        pass
