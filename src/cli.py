from pathlib import Path
from typing import Optional


from src.dataset_searcher import DatasetSearcher
from src.evaluater import Evaluater
from src.indexer.indexer import Indexer
from src.indexer.incremental_indexer import IncrementalIndexer
from src.utils.io_utils import read_json_model, write_json_model
from src.models import (
    RagDataset, StudentSearchResults, StudentSearchResultsAndAnswer,
    MinimalSearchResults, MinimalSource, UnansweredQuestion,
)
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.exceptions import IndexingError
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
    """Expose indexing, retrieval, generation and evaluation commands."""

    def __init__(self) -> None:
        """Create the lexical retriever and CLI input validator."""
        self._retriever = BM25Retriever()
        self._validator = Validator()

    def index(
        self, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        incremental: bool = False, data_path: str = DEFAULT_RAW_DATA_PATH,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Build and save a full or incremental corpus index."""
        max_chunk_size = self._validator.validate_chunk_size_value(
            max_chunk_size)

        if not isinstance(incremental, bool):
            raise IndexingError(
                f"--incremental takes no value, got {incremental!r}")
        if incremental:
            indexer: Indexer = IncrementalIndexer(
                max_chunk_size=max_chunk_size,
                data_path=data_path,
                retriever=self._retriever,
                index_path=index_path
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

    def search(
        self, query: str, k: int = DEFAULT_TOP_K,
        index_path: str = DEFAULT_INDEX_PATH,
        output_path: str = DEFAULT_SEARCH_RESULTS_DIRECTORY + "/query.json",
    ) -> None:
        """Search one question and save validated JSON results."""
        question = UnansweredQuestion(question=str(query))
        self._retriever.load(index_path)
        sources = self._retriever.retrieve(query=question.question, k=k)
        result = MinimalSearchResults(
            question_id=question.question_id, question=question.question,
            retrieved_sources=[
                MinimalSource.model_validate(source) for source in sources],
        )
        results = StudentSearchResults(search_results=[result], k=k)
        write_json_model(output_path, results)
        print(results.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = DEFAULT_TOP_K,
        save_directory: Optional[str] = None,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Retrieve sources for every question and save the results JSON."""
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
        self, query: str, k: int = DEFAULT_TOP_K,
        index_path: str = DEFAULT_INDEX_PATH,
        output_path: str = DEFAULT_ANSWERS_DIRECTORY + "/query.json",
    ) -> None:
        """Answer one question through RagService and save its JSON."""
        from src.rag_service import RagService

        result = RagService(index_path=index_path).answer_result(
            query=str(query), k=k
        )
        results = StudentSearchResultsAndAnswer(search_results=[result], k=k)
        write_json_model(output_path, results)
        print(results.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: Optional[str] = None,
    ) -> None:
        """Generate and save answers from a saved search results dataset."""
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
        """Compute retrieval recall against the reference question dataset."""
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
