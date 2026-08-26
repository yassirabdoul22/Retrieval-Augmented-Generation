from abc import ABC, abstractmethod
from typing import List
from src.models import Chunk


# this Abstract class is the base of strategy pattern
# it will be extended by Codechunker/MarkdownChunker


class Chunker(ABC):
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, file_path: str, content: str) -> List[Chunk]:
        units = self._parse(file_path,content)
        return self._enforce_max_size(units)

    @abstractmethod
    def _parse(self,file_path, content: str) -> List[Chunk]:
        pass

    def _enforce_max_size(self, units: List[Chunk]) -> List[Chunk]:
        enforced_units: List[Chunk] = []
        for unit in units:
            if len(unit.text) > self.max_chunk_size:
                start = unit.first_character_index
                local_start = 0
                while local_start < len(unit.text):
                    piece = unit.text[
                        local_start:local_start + self.max_chunk_size
                    ]
                    end = start + len(piece)
                    new_unit = Chunk(
                        file_path=unit.file_path,
                        first_character_index=start,
                        last_character_index=end,
                        text=piece,
                    )
                    enforced_units.append(new_unit)
                    start = end
                    local_start += self.max_chunk_size
            else:
                enforced_units.append(unit)
        return enforced_units

    def compute_line_offsets(self, content: str) -> List[int]:
        lines = content.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        return offsets