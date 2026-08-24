from typing import List

from .base import Chunker


class MarkdownChunker(Chunker):

    def _parse(self, content: str) -> List[str]:
        raise NotImplementedError
