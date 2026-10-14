from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError
from src.utils.exceptions import IndexingError
from src.models import MinimalSource





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


def read_source_text(source: MinimalSource) -> str:
    """Re-read a source's raw text from disk, using its character range."""
    try:
        with open(source.file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except (OSError, UnicodeDecodeError) as e:
        raise IndexingError(
            f"cannot read source file {source.file_path}"
        ) from e
    return content[source.first_character_index:source.last_character_index]
