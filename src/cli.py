from pathlib import Path
from typing import Optional

import fire

from src.dataset_searcher import DatasetSearcher
from src.evaluater import Evaluater
from src.indexer.indexer import Indexer
from src.indexer.incremental_indexer import IncrementalIndexer
from src.utils.io_utils import read_json_model, write_json_model
from src.models import RagDataset, StudentSearchResults
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.io_utils import read_source_text
from src.utils.exceptions import IndexingError, IndexPersistenceError
from src.utils.validator import Validator
from src.utils.config import (
    DEFAULT_RAW_DATA_PATH,
    DEFAULT_INDEX_PATH,
    DEFAULT_SEARCH_RESULTS_DIRECTORY,
    DEFAULT_ANSWERS_DIRECTORY,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_TOP_K
)
class CLI:

    def __init__(self) -> None:
        self._retriever = BM25Retriever()
        self._validator = Validator()

    def index(self, max_chunk_size:int = DEFAULT_MAX_CHUNK_SIZE, incremental:bool=False, data_path: str = DEFAULT_RAW_DATA_PATH, index_path: str = DEFAULT_INDEX_PATH) -> None:
        max_chunk_size = self._validator.validate_chunk_size_value(max_chunk_size)
        
        if not isinstance(incremental,bool):
            raise IndexingError(f"--incremental takes no value, got {incremental!r}")        
        if incremental:
            indexer: Indexer =IncrementalIndexer(
                max_chunk_size= max_chunk_size,
                data_path= data_path,
                retriever= self._retriever,
                index_path= index_path
            )
        else:
            indexer = Indexer(
                data_path=data_path,
                max_chunk_size=max_chunk_size,
                retriever=self._retriever,
            )
        
        indexer.indexing()
        indexer.save(index_path)
        print(f"Ingestion complete! Indices saved under {index_path}")

    def search(self, query: str, k:int = DEFAULT_TOP_K,index_path =DEFAULT_INDEX_PATH) -> None:
        query = str(query)
        try:
            self._retriever.load(index_path)
            k = self._validator.validate_top_k_value(k,len(self._retriever.chunks))
        except IndexPersistenceError:
            print("Error: no index found. Run the index command first:")
            print(f"uv run python -m src index --max_chunk_size {DEFAULT_MAX_CHUNK_SIZE}")
            exit(1)
        results = self._retriever.retrieve(query=query, k=k)
        if not results:
            print("No relevant source found for this query.")
            return
        for r in results:
            print(
                f"{r.file_path} "
                f"[{r.first_character_index}:{r.last_character_index}]"
            )
                    
    def search_dataset(
        self,
        dataset_path: str,
        k: int = DEFAULT_TOP_K,
        save_directory: Optional[str] = None,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        dataset = read_json_model(dataset_path, RagDataset)
        self._retriever.load(index_path)
        results = DatasetSearcher(retriever=self._retriever, k=k).search(
            dataset=dataset
        )
        output_path = self._output_path(
            dataset_path, save_directory, DEFAULT_SEARCH_RESULTS_DIRECTORY
        )
        write_json_model(output_path, results)
        print(f"Saved student_search_results to {output_path}")

    def answer(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        from src.rag_service import RagService
        print(RagService(index_path=index_path).answer(query=query, k=k))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: Optional[str] = None,
    ) -> None:
        from src.dataset_answerer import DatasetAnswerer
        from src.generator import Generator
        search_results = read_json_model(
            student_search_results_path, StudentSearchResults
        )
        generator = Generator()
        results = DatasetAnswerer(generator).answer(search_results)
        output_path = self._output_path(
            student_search_results_path,
            save_directory,
            DEFAULT_ANSWERS_DIRECTORY,
        )
        write_json_model(output_path, results)
        print(f"Saved student_search_results_and_answer to {output_path}")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        student_results = read_json_model(
            student_search_results_path, StudentSearchResults
        )
        reference = read_json_model(dataset_path, RagDataset)
        result = Evaluater().evaluate(student_results, reference)
        for k, v in result.items():
            print(f"{k}: {v:.3f} ({v * 100:.1f}%)")

    @staticmethod
    def _output_path(
        input_path: str, save_directory: Optional[str], default_root: str
    ) -> str:
        """Return where to save the output of input_path.

        Without --save_directory, the output goes under default_root,
        scoped by the input's folder (e.g. UnansweredQuestions) so runs
        on datasets sharing a file name do not overwrite each other.
        """
        source = Path(input_path)
        if save_directory is None:
            directory = Path(default_root) / source.parent.name
        else:
            directory = Path(str(save_directory))
        return str(directory / source.name)
