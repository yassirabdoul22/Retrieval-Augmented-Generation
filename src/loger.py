from enum import Enum


class LogerType(Enum):
    UNSUPPORTED_EXTENSION = "UNSUPPORTED EXTENSION"
    FILE_READ_ERROR = "FILE READ ERROR"
    INVALID_SYNTAX = "INVALID SYNTAX"


class Loger:
    def __init__(self, loger_type: LogerType):
        self.loger_type = loger_type

    def log(self, warning: str) -> None:
        print(f"[{self.loger_type.value}]: {warning}")
