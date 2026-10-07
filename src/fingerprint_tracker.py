"""Fingerprints of corpus files, used to re-index only what changed."""

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

BLOCK_SIZE = 1 << 16


class FileFingerPrint(BaseModel):
    """Identity of one file's content at indexing time."""

    mtime_ns: int
    size: int
    md5: str


class FingerPrintSnapshot(BaseModel):
    """What the saved index was built from."""

    max_chunk_size: int
    files: Dict[str, FileFingerPrint] = Field(default_factory=dict)


class FingerprintTracker:
    """Loads, saves and compares snapshots of corpus fingerprints."""

    def __init__(self, snapshot_path: str) -> None:
        """Remember where the snapshot JSON file lives."""
        self._snapshot_path = Path(snapshot_path)

    @staticmethod
    def compute_md5(file_path: Path) -> str:
        """Return the MD5 of a file, read in 64 KiB blocks."""
        digest = hashlib.md5(usedforsecurity=False)
        with open(file_path, "rb") as file:
            while buf := file.read(BLOCK_SIZE):
                digest.update(buf)
        return digest.hexdigest()

    def load(self) -> Optional[FingerPrintSnapshot]:
        """Return the saved snapshot, or None if absent or unreadable."""
        try:
            return FingerPrintSnapshot.model_validate_json(
                self._snapshot_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            return None

    def save(self, snapshot: FingerPrintSnapshot) -> None:
        """Write the snapshot as JSON, creating the folder if needed."""
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshot_path.write_text(
            snapshot.model_dump_json(), encoding="utf-8"
        )

    @classmethod
    def detect_changes(
        cls, old: FingerPrintSnapshot, files: Iterable[Path]
    ) -> Tuple[Set[str], Set[str], Dict[str, FileFingerPrint]]:
        """Compare files on disk with the previous snapshot.

        Returns:
            (to_chunk, deleted, finger_prints): files to (re)chunk, files
            that disappeared, and the fingerprints to save next.
        """
        to_chunk: Set[str] = set()
        finger_prints: Dict[str, FileFingerPrint] = {}
        for path in files:
            key = str(path)
            try:
                file_metadata = path.stat()
            except FileNotFoundError:
                continue
            prev = old.files.get(key)
            if (prev is not None
                    and prev.mtime_ns == file_metadata.st_mtime_ns
                    and prev.size == file_metadata.st_size):
                finger_prints[key] = prev
                continue
            md5 = cls.compute_md5(path)
            if prev is None or prev.md5 != md5:
                to_chunk.add(key)
            finger_prints[key] = FileFingerPrint(
                mtime_ns=file_metadata.st_mtime_ns,
                size=file_metadata.st_size,
                md5=md5,
            )
        deleted = set(old.files) - set(finger_prints)
        return to_chunk, deleted, finger_prints