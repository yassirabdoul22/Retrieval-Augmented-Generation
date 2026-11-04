"""Decorator that adds a query-result cache in front of any retriever."""

import json
import shutil
from hashlib import md5
from pathlib import Path
from typing import List, Optional

from src.indexer.fingerprint_tracker import FingerprintTracker
from src.models import Chunk
from src.retrieval.base import Retriever
from src.utils.config import CACHE_DIR, CHUNKS_FILE, INDEX_HASH_FILE
from src.utils.exceptions import RetrievalError
from src.utils.loger import Loger, LogerType


class RetrieverCache(Retriever):
    """Wrap a retriever and cache the results of its queries."""

    def __init__(self, retriever: Retriever) -> None:
        """Keep the wrapped retriever; the cache folder is set by load()."""
        self._retriever = retriever
        self._cache_dir: Optional[Path] = None
        self._index_hash: Optional[str] = None

    @property
    def chunks(self) -> List[Chunk]:
        """Expose the wrapped retriever's chunks (used by --incremental)."""
        return self._retriever.chunks

    def index(self, chunks: List[Chunk]) -> None:
        """Index the chunks with the wrapped retriever."""
        self._retriever.index(chunks)

    def load(self, path: str) -> None:
        """Load the wrapped index, its cache folder and its hash."""
        self._retriever.load(path)
        self._cache_dir = Path(path) / CACHE_DIR
        try:
            with open(Path(path) / INDEX_HASH_FILE, encoding="utf-8") as file:
                self._index_hash = file.read().strip()
        except OSError:
            Loger(LogerType.GENERAL_ERROR).log(
                "impossible to load index fingerprint: "
                "invalid format or missing"
            )
            self._index_hash = None

    def save(self, path: str) -> None:
        """Save the wrapped index, drop the old cache, store the new hash."""
        self._retriever.save(path)
        shutil.rmtree(Path(path) / CACHE_DIR, ignore_errors=True)
        index_hash = FingerprintTracker.compute_md5(Path(path) / CHUNKS_FILE)
        try:
            with open(Path(path) / INDEX_HASH_FILE, "w",
                      encoding="utf-8") as file:
                file.write(index_hash)
        except OSError as e:
            raise RetrievalError(f"index hash not saved: {e}") from e

    def retrieve(self, query: str, k: int) -> List[Chunk]:
        """Return the top-k chunks using cache if it's possible"""
        if self._cache_dir is None or self._index_hash is None:
            return self._retriever.retrieve(query=query, k=k)

        key = "|".join([str(self._index_hash), str(k), query])

        name = md5(key.encode("utf-8"), usedforsecurity=False).hexdigest()

        cache_file = self._cache_dir / f"{name}.json"

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_chunks = [Chunk(**d) for d in data]
            print(f"response from cached result of prompt {query}")
            return cached_chunks
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            Loger(LogerType.GENERAL_ERROR).log(
                f"corrupted cache entry ignored: {cache_file}"
            )
        chunks = self._retriever.retrieve(query=query, k=k)

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps([c.model_dump() for c in chunks]),
                encoding="utf-8"
            )
            print("result cached by succes!!!!!!!!!!!!!!!")
        except OSError:
            Loger(LogerType.GENERAL_ERROR).log("query result not cached")

        return chunks
