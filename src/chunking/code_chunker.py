from typing import List

import ast

from src.models import Chunk 
from .base import Chunker
from src.exceptions import InvalidPythonSyntaxeError

class CodeChunker(Chunker):

    def _parse(self, file_path: str, content: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        try:
            content_tree = ast.parse(content)
        except SyntaxError as e:
            raise InvalidPythonSyntaxeError(f"{file_path} : {e} ") from e
        offsets = self.compute_line_offsets(content=content)
        waiting_nodes: List[ast.AST] = []

        for node in ast.iter_child_nodes(content_tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if waiting_nodes:
                    chunks.append(
                        self._group_nodes(
                            file_path, content, offsets, waiting_nodes
                        )
                    )
                    waiting_nodes = []
                start = offsets[node.lineno - 1]
                end = offsets[node.end_lineno]
                chunk_content = content[start:end]
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end,
                        text=chunk_content,
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
        waiting_nodes: List[ast.AST],
    ) -> Chunk:
        start = offsets[waiting_nodes[0].lineno - 1]
        end = offsets[waiting_nodes[-1].end_lineno]
        return Chunk(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            text=content[start:end],
        )
