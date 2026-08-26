from typing import List
from src.models import Chunk
from .base import Chunker


class MarkdownChunker(Chunker):

    def _parse(self, content: str) -> List[Chunk]:
        raise NotImplementedError
