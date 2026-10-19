from src.utils.exceptions import InvalidChunkSizeError, InvalidTopKError


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
