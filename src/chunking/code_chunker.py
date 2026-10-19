"""Chunking strategy for Python source files, based on the ``ast`` module."""

import ast
from typing import List

from src.utils.exceptions import InvalidPythonSyntaxeError
from src.models import Chunk

from .base import Chunker


class CodeChunker(Chunker):
    """Splits a Python file into chunks using its top-level structure."""

    def _parse(self, file_path: str, content: str) -> List[Chunk]:
        """Chunk each top-level function/class, grouping the rest."""
        chunks: List[Chunk] = []
        try:
            content_tree = ast.parse(content)
        except SyntaxError as e:
            raise InvalidPythonSyntaxeError(f"{file_path}: {e}") from e

        offsets = self.compute_line_offsets(content)
        start = 0
        for node in content_tree.body:
            if not isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
            )):
                continue
            assert node.end_lineno is not None
            end = offsets[node.end_lineno]
            chunks.append(Chunk(
                file_path=file_path, first_character_index=start,
                last_character_index=end, text=content[start:end],
            ))
            start = end
        if start < len(content):
            chunks.append(Chunk(
                file_path=file_path, first_character_index=start,
                last_character_index=len(content), text=content[start:],
            ))
        return chunks
