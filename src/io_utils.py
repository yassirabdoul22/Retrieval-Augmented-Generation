from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.exceptions import IndexingError

T = TypeVar("T", bound=BaseModel)


def read_json_model(path: str, model: Type[T]) -> T:
    try:
        with open(path) as f:
            return model.model_validate_json(f.read())
    except FileNotFoundError as e:
        raise IndexingError(f"dataset not found: {path}") from e
    except ValidationError as e:
        raise IndexingError(f"malformed dataset: {path}: {e}") from e


def write_json_model(path: str, model: BaseModel) -> None:
    try:
        with open(path, "w") as file:
            file.write(model.model_dump_json(indent=2))
    except OSError as e:
        raise IndexingError(f"cannot write to {path}: {e}") from e
