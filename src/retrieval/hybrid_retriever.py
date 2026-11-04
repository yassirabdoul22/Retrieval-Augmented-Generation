from typing import Dict, List, Tuple

from src.models import Chunk
from src.retrieval.base import Retriever
from src.utils.config import HYBRID_CANDIDATES, RRF_CONSTANT
from src.utils.validator import Validator

ChunkKey = Tuple[str, int, int]


class HybridRetriever(Retriever):
    def __init__(self, lexical: Retriever, sementic: Retriever) -> None:
        self._lexical = lexical
        self._sementic = sementic
        self._validator = Validator()

    def index(self, chunks: List[Chunk]) -> None:
        self._lexical.index(chunks)
        self._sementic.index(chunks)

    def save(self, path: str) -> None:
        self._lexical.save(path)
        self._sementic.save(path)

    def load(self, path: str) -> None:
        self._lexical.load(path)
        self._sementic.load(path)

    @property
    def chunks(self) -> List[Chunk]:
        return self._lexical.chunks

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        k = self._validator.validate_top_k_value(k, len(self.chunks))
        n = min(max(k, HYBRID_CANDIDATES), len(self.chunks))
        sementic_chunks = self._sementic.retrieve(query, n)
        lexical_chunks = self._lexical.retrieve(query, n)
        rrf_scores, chunk_by_key = self.chunks_by_score(
            sementic_chunks, lexical_chunks
        )
        best = sorted(
            rrf_scores, key=lambda k: rrf_scores[k], reverse=True
        )[:k]
        return [chunk_by_key[key] for key in best]

    def chunks_by_score(
        self, sementic_chunks: List[Chunk], lexical_chunks: List[Chunk]
    ) -> Tuple[Dict[ChunkKey, float], Dict[ChunkKey, Chunk]]:
        scores: Dict[ChunkKey, float] = {}
        chunk_by_key: Dict[ChunkKey, Chunk] = {}
        for results in (sementic_chunks, lexical_chunks):
            for rank, chunk in enumerate(results, start=1):
                key = (chunk.file_path, chunk.first_character_index,
                       chunk.last_character_index)
                scores[key] = scores.get(key, 0.0) + 1 / (RRF_CONSTANT + rank)
                chunk_by_key[key] = chunk
        return scores, chunk_by_key
