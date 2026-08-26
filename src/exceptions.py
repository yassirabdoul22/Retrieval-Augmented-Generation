

class ChunkingError(Exception):
    def __init__(self, message: str):
        super().__init__(f"[Error] {message}")


class UnvailableChunkerError(ChunkingError):
    pass

class InvalidPythonSyntaxeError(ChunkingError):
    pass