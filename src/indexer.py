from pathlib import Path
from typing import List

from src.chunking import get_chunker
from src.exceptions import (
    IndexingError,
    InvalidPythonSyntaxeError,
    UnvailableChunkerError,
)
from src.loger import Loger, LogerType
from src.models import Chunk
from src.retrieval.base import Retriever


class Indexer:

    def __init__(
        self, data_path: str, max_chunk_size: int, retriever: Retriever
    ) -> None:
        self.data_path = data_path
        self.max_chunk_size = max_chunk_size
        self._directory = Path(data_path)
        self.chunks: List[Chunk] = []
        self.retriever = retriever

    def indexing(self) -> None:
        self.load_chunks()
        self.retriever.index(self.chunks)

    def load_chunks(self) -> None:
        for file_path in self._directory.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                chunker = get_chunker(
                    file_path=str(file_path),
                    max_chunk_size=self.max_chunk_size,
                )
            except UnvailableChunkerError:
                Loger(LogerType.UNSUPPORTED_EXTENSION).log(str(file_path))
                continue
            try:
                content = self._get_file_content(file_path=str(file_path))
                self.chunks.extend(
                    chunker.chunk(file_path=str(file_path), content=content)
                )
            except IndexingError:
                Loger(LogerType.FILE_READ_ERROR).log(str(file_path))
                continue
            except InvalidPythonSyntaxeError:
                Loger(LogerType.INVALID_SYNTAX).log(str(file_path))
                continue

    def _get_file_content(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except (OSError, UnicodeDecodeError) as e:
            raise IndexingError(f"cannot read file {file_path}") from e
