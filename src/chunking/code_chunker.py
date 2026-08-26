"""Chunking strategy for Python source files, based on the ``ast`` module."""

import ast
from typing import List

from src.exceptions import InvalidPythonSyntaxeError
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

        offsets = self.compute_line_offsets(content=content)
        waiting_nodes: List[ast.stmt] = []

        for node in content_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if waiting_nodes:
                    chunks.append(
                        self._group_nodes(
                            file_path, content, offsets, waiting_nodes
                        )
                    )
                    waiting_nodes = []
                assert node.end_lineno is not None
                start = offsets[node.lineno - 1]
                end = offsets[node.end_lineno]
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end,
                        text=content[start:end],
                    )
                )
            else:
                waiting_nodes.append(node)

        if waiting_nodes:
            chunks.append(
                self._group_nodes(file_path, content, offsets, waiting_nodes)
            )

        return chunks

    def _group_nodes(
        self,
        file_path: str,
        content: str,
        offsets: List[int],
        waiting_nodes: List[ast.stmt],
    ) -> Chunk:
        """Merge consecutive non-function/class nodes into one chunk."""
        last_node = waiting_nodes[-1]
        assert last_node.end_lineno is not None
        start = offsets[waiting_nodes[0].lineno - 1]
        end = offsets[last_node.end_lineno]
        return Chunk(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            text=content[start:end],
        )
