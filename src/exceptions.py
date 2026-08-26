"""Exception hierarchy for the chunking module."""


class ChunkingError(Exception):
    """Base exception for all chunking-related failures."""

    def __init__(self, message: str):
        super().__init__(f"[Error] {message}")


class UnvailableChunkerError(ChunkingError):
    """Raised when no chunker is available for a given file extension."""


class InvalidPythonSyntaxeError(ChunkingError):
    """Raised when a Python file cannot be parsed by ``ast``."""
