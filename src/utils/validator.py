from src.utils.exceptions import (
    InvalidChunkSizeError, InvalidPromptError, InvalidTopKError,
    RetrievalError,
)


class Validator:
    """Validate chunk sizes and retrieval result counts."""
    MAX_CHUNK_SIZE = 2000

    def validate_chunk_size_value(self, chunk_size: int) -> int:
        """Require an integer chunk size between 1 and 2000 characters."""
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise InvalidChunkSizeError(
                f"max_chunk_size must be an integer, got {chunk_size!r}")

        if not 1 <= chunk_size <= Validator.MAX_CHUNK_SIZE:
            raise InvalidChunkSizeError(
                "chunk size must be grater the 0"
                f"and less then {Validator.MAX_CHUNK_SIZE}"
            )
        return chunk_size

    def validate_top_k_value(self, k: object, chunks_len: int) -> int:
        """Require a positive integer k within the available chunk count."""
        if isinstance(k, bool) or not isinstance(k, int):
            raise InvalidTopKError(f"k must be an integer go {k!r}")
        if k <= 0:
            raise InvalidTopKError(f"k must be a positive integer got {k!r}")
        if k > chunks_len:
            raise InvalidTopKError(
                f"impossible rank {k} chunks (there is {chunks_len!r})")
        return k

    def validate_prompt(self, query: object) -> str:
        """Strip the query and reject it if empty or whitespace-only."""
        text = str(query).strip()
        if not text:
            raise InvalidPromptError("query must not be empty")
        return text

    def validate_use_cache(self, use_cache: object) -> bool:
        """Reject a value passed to --use_cache, which is a plain flag."""
        if not isinstance(use_cache, bool):
            raise RetrievalError(
                f"--use_cache takes no value, got {use_cache!r}")
        return use_cache

    def validate_hybrid(self, hybrid: object) -> bool:
        """Reject a value passed to --hybrid, which is a plain flag."""
        if not isinstance(hybrid, bool):
            raise RetrievalError(f"--hybrid takes no value, got {hybrid!r}")
        return hybrid
