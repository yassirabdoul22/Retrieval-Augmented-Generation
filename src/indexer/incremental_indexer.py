"""Incremental indexing: re-chunk only the files that changed (bonus)."""

import os
from pathlib import Path
from typing import List, Optional

from src.models import Chunk
from src.retrieval.base import Retriever
from src.utils.exceptions import IndexPersistenceError, UnvailableChunkerError, IndexingError, InvalidPythonSyntaxeError
from src.utils.loger import Loger, LogerType
from src.chunking import get_chunker

from .fingerprint_tracker import (
    FINGERPRINTS_FILE,
    FingerprintTracker,
    FingerPrintSnapshot,
)
from .indexer import Indexer


class IncrementalIndexer(Indexer):
    """Keeps the chunks of unchanged files and re-chunks the others."""

    def __init__(
        self,
        data_path: str,
        max_chunk_size: int,
        retriever: Retriever,
        index_path: str,
    ) -> None:
        """Set up the indexer, the previous index location and the tracker."""
        super().__init__(data_path, max_chunk_size, retriever)
        self._index_path = index_path
        self._tracker = FingerprintTracker(
            os.path.join(index_path, FINGERPRINTS_FILE)
        )
        self._snapshot = FingerPrintSnapshot(max_chunk_size=max_chunk_size)

    def indexing(self) -> None:
        """Re-chunk only changed files, keep the others, rebuild BM25."""
        snapshot = self._tracker.load()
        old_chunks = self._load_old_chunks()

        reason = self._full_re_indexing_reason(snapshot, old_chunks)
        if reason is not None or snapshot is None:
            Loger(LogerType.FULL_REINDEXING).log(reason or "no snapshot")
            self._snapshot = self._fingerprint_everything()
            super().indexing()
            return

        to_chunk, deleted, prints = self._tracker.detect_changes(
            snapshot, self._corpus_files()
        )
        stale = to_chunk | deleted
        kept = [c for c in old_chunks if c.file_path not in stale]
        fresh = self._chunk_files(Path(p) for p in to_chunk)
        self.chunks = kept + fresh
        self.retriever.index(self.chunks)
        self._snapshot = FingerPrintSnapshot(
            max_chunk_size=self.max_chunk_size, files=prints
        )
        print(f"Re-chunked {len(to_chunk)} file(s), removed {len(deleted)}, "
              f"total {len(self.chunks)} chunks")

    def save(self, index_path: str) -> None:
        """Store the index first, the fingerprints last."""
        super().save(index_path)
        self._tracker.save(self._snapshot)

    def _load_old_chunks(self) -> List[Chunk]:
        """Return the chunks of the previous index, or [] if unusable."""
        try:
            self.retriever.load(self._index_path)
        except IndexPersistenceError:
            return []
        return list(self.retriever.chunks)

    def _fingerprint_everything(self) -> FingerPrintSnapshot:
        """Fingerprint every corpus file, as after a full rebuild."""
        empty = FingerPrintSnapshot(max_chunk_size=self.max_chunk_size)
        _, _, prints = self._tracker.detect_changes(
            empty, self._corpus_files()
        )
        return FingerPrintSnapshot(
            max_chunk_size=self.max_chunk_size, files=prints
        )

    def _full_re_indexing_reason(
        self,
        snapshot: Optional[FingerPrintSnapshot],
        old_chunks: List[Chunk],
    ) -> Optional[str]:
        """Explain why the index must be rebuilt from scratch, or None."""
        if snapshot is None:
            return "no previous fingerprints (first run or corrupted file)"
        if not old_chunks:
            return "no previous index to update"
        if snapshot.max_chunk_size != self.max_chunk_size:
            return (f"max_chunk_size changed "
                    f"({snapshot.max_chunk_size} -> {self.max_chunk_size})")
        return None
    
    def _corpus_files(self) -> List[Path]:
        """Return the list of all files in the corpus."""
        return [p for p in Path(self.data_path).rglob("*") if p.is_file()]

    
    
    def _chunk_files(self, files: List[Path]) -> List[Chunk]:
        chunks = []
        for file in files:
            try:
                chunker = get_chunker(str(file),self.max_chunk_size)
            except UnvailableChunkerError:
                Loger(LogerType.UNSUPPORTED_EXTENSION).log(str(file))
                continue
            try:
                content = self._get_file_content(str(file))
                chunks.extend(chunker.chunk(str(file),content))
            except IndexingError:
                Loger(LogerType.FILE_READ_ERROR).log(str(file))
            except InvalidPythonSyntaxeError:
                Loger(LogerType.INVALID_SYNTAX).log(str(file))
        
        return chunks
