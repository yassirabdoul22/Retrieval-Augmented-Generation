"""Chunking strategy for Markdown files, based on heading boundaries."""

import re
from typing import List

from src.models import Chunk

from .base import Chunker

HEADER_PATTERN = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


class MarkdownChunker(Chunker):
    """Splits a Markdown file into sections delimited by headings."""

    def _parse(self, file_path: str, content: str) -> List[Chunk]:
        """Chunk each section between two headings."""
        chunks: List[Chunk] = []
        headers = list(HEADER_PATTERN.finditer(content))

        if not headers:
            return [
                Chunk(
                    file_path=file_path,
                    first_character_index=0,
                    last_character_index=len(content),
                    text=content,
                )
            ]

        if headers[0].start() > 0:
            intro_end = headers[0].start()
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=0,
                    last_character_index=intro_end,
                    text=content[0:intro_end],
                )
            )

        for i in range(len(headers)):
            start = headers[i].start()
            if i + 1 < len(headers):
                end = headers[i + 1].start()
            else:
                end = len(content)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=content[start:end],
                )
            )
        return chunks
