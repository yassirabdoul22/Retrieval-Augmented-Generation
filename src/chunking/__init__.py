"""Factory that selects the right Chunker strategy for a given file."""

import re

from src.exceptions import UnvailableChunkerError

from .base import Chunker
from .code_chunker import CodeChunker
from .markdown_chunker import MarkdownChunker

EXTENSION_PATTERN = re.compile(r"\.([a-zA-Z0-9]+)$")


def get_chunker(file_path: str, max_chunk_size: int) -> Chunker:
    """Return the Chunker matching a file's extension."""
    extension_match = EXTENSION_PATTERN.search(file_path)
    extension = extension_match.group(1) if extension_match else None
    match extension:
        case "py":
            return CodeChunker(max_chunk_size=max_chunk_size)
        case "md":
            return MarkdownChunker(max_chunk_size=max_chunk_size)
        case None:
            raise UnvailableChunkerError(f"{file_path} has no file extension")
        case _:
            raise UnvailableChunkerError(f"{extension} is not supported")
