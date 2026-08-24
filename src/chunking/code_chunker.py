from typing import List

from .base import Chunker


class CodeChunker(Chunker):

    def _parse(self, content: str) -> List[str]:
        raise NotImplementedError
