"""Exception hierarchy for the chunking, retrieval and indexing modules."""


class PipelineError(Exception):
    """Base exception for failures reported by the RAG pipeline."""

    def __init__(self, message: str) -> None:
        """Prefix pipeline error messages consistently."""
        super().__init__(f"[Error] {message}")


class ChunkingError(PipelineError):
    """Base exception for all chunking-related failures."""


class UnvailableChunkerError(ChunkingError):
    """Raised when no chunker is available for a given file extension."""


class InvalidPythonSyntaxeError(ChunkingError):
    """Raised when a Python file cannot be parsed by ``ast``."""


class InvalidChunkSizeError(ChunkingError):
    """Raised when invalid chunk size value is passed"""


class RetrievalError(PipelineError):
    """Base exception for all retrieval-related failures."""


class InvalidTopKError(RetrievalError):
    """ Raised when k is less or equal 0"""


class InvalidPromptError(RetrievalError):
    """Raised when the query is empty or contains only whitespace."""


class IndexPersistenceError(RetrievalError):
    """Raised when a retriever's index cannot be saved or loaded."""


class IndexingError(PipelineError):
    """Base exception for all indexing-related failures."""
