from abc import ABC, abstractmethod
from typing import List
from src.models import Chunk


# this Abstract class is the base of strategy pattern
# it will be extended by Codechunker/MarkdownChunker


class Chunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, content: str) -> List[Chunk]:
        units = self._parse(content)
        return self._enforce_max_size(units, file_path)

    @abstractmethod
    def _parse(self, content: str) -> List[str]:
        pass

    def _enforce_max_size(
        self, units: List[str], file_path: str
    ) -> List[Chunk]:
        raise NotImplementedError
