from .base import Chunker
from .code_chunker import CodeChunker
from .markdown_chunker import MarkdownChunker
from src.exceptions import UnvailableChunkerError
import re


def get_chunker(file_path: str, max_chunk_size: int) -> Chunker:
    extention_match = re.search(r"\.([a-zA-Z0-9]+)$", file_path)
    extention = extention_match.group(1) if extention_match else None
    match extention:
        case "py":
            return CodeChunker(max_chunk_size=max_chunk_size)
        case "md":
            return MarkdownChunker(max_chunk_size=max_chunk_size)
        case None:
            raise UnvailableChunkerError(f"{file_path} has no file extension")
        case _:
            raise UnvailableChunkerError(f"{extention} is not supported")
