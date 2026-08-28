"""Exception hierarchy for the chunking, retrieval and indexing modules."""


class ChunkingError(Exception):
    """Base exception for all chunking-related failures."""

    def __init__(self, message: str):
        super().__init__(f"[Error] {message}")


class UnvailableChunkerError(ChunkingError):
    """Raised when no chunker is available for a given file extension."""


class InvalidPythonSyntaxeError(ChunkingError):
    """Raised when a Python file cannot be parsed by ``ast``."""


class RetrievalError(Exception):
    """Base exception for all retrieval-related failures."""

    def __init__(self, message: str):
        super().__init__(f"[Error] {message}")


class IndexPersistenceError(RetrievalError):
    """Raised when a retriever's index cannot be saved or loaded."""


class IndexingError(Exception):
    """Base exception for all indexing-related failures."""

    def __init__(self, message: str):
        super().__init__(f"[Error] {message}")


class IndexedFileError(IndexingError):
    """Raised when a file cannot be read during indexing."""
