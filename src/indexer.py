from src.chunking import get_chunker
from src.chunking.base import Chunker
from src.retrieval.base import Retriever
from src.chunking.code_chunker import CodeChunker
from src.chunking.markdown_chunker import MarkdownChunker
from pathlib import Path
from typing import List
from models import Chunk
from .exceptions import IndexingError


class Indexer:

    def __int__(self , data_path:str , max_chunk_size: int, retriever:Retriever):
        self.data_path= data_path
        self.max_chunk_size= max_chunk_size
        self._directory= Path(data_path)
        self.chunks :List[Chunk] = []
        self.retriever= retriever

    def indexing(self):
        self.load_chunks()
        self.retriever.retrieve(self.chunks)

    def load_chunks(self):
        for file_path in  self._directory.rglob("*"):
            chunker = get_chunker(self.max_chunk_size)
            self.chunks.extend(chunker.chunk(content = self._get_file_content(file_path=file_path) , file_path=file_path))


    def _get_file_content(self , file_path: str):
        try:
            with open(file_path , "r" , encoding="utf-8") as file:
                return file.read()
        except OSError as e:
            raise IndexingError(f"cannot read file {file_path}") from e 