from .indexer import Indexer
from fingerprint_tracker import FingerPrintSnapshot
from typing import Optional, List
from src.models import Chunk
from src.utils.loger import Loger

class IncrementalIndexer(Indexer):


    def _full_re_indexing_reason(self,snapshot:FingerPrintSnapshot,old_chunks:List[Chunk])->Optional[str]:
        if snapshot is None:
            return "no prev finger print (first run or corrupted file)"
        if not old_chunks:
            return "no prev index to update"
        if snapshot.max_chunk_size != self.max_chunk_size:
            return "max_chunk size changed"
        return None

    