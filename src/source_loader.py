from src.exceptions import IndexingError
from src.models import MinimalSource


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
