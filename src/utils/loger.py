# log level (loger type)


from enum import Enum


class LogerType(Enum):
    """Categories of indexing diagnostics."""
    GENERAL_LOGER = "LOGER"
    UNSUPPORTED_EXTENSION = "UNSUPPORTED EXTENSION"
    FILE_READ_ERROR = "FILE READ ERROR"
    INVALID_SYNTAX = "INVALID SYNTAX"
    FULL_REINDEXING = "FULL REINDEXING"


class Loger:
    """Print categorized indexing diagnostics."""

    def __init__(self, loger_type: LogerType):
        """Store the diagnostic category used to prefix messages."""
        self.loger_type = loger_type

    def log(self, warning: str) -> None:
        """Print a diagnostic message prefixed with its category."""
        print(f"[{self.loger_type.value}]: {warning}")
