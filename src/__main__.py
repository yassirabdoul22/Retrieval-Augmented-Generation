from pathlib import Path

import fire

from src.dataset_answerer import DatasetAnswerer
from src.dataset_searcher import DatasetSearcher
from src.evaluater import Evaluater
from src.generator import Generator
from src.indexer import Indexer
from src.io_utils import read_json_model, write_json_model
from src.models import RagDataset, StudentSearchResults
from src.retrieval.bm25_retriever import BM25Retriever
from src.source_loader import read_source_text


class CLI:

    def __init__(self) -> None:
        self._retriever = BM25Retriever()

    def index(self, max_chunk_size: int = 2000) -> None:
        indexer = Indexer(
            data_path="data/raw",
            max_chunk_size=max_chunk_size,
            retriever=self._retriever,
        )
        indexer.indexing()
        self._retriever.save("data/processed")

    def search(self, query: str, k: int = 10) -> None:
        query = str(query)
        self._retriever.load("data/processed")
        results = self._retriever.retrieve(query=query, k=k)
        for r in results:
            print(
                f"{r.file_path} "
                f"[{r.first_character_index}:{r.last_character_index}]"
            )

    def search_dataset(
        self, dataset_path: str, k: int = 10, save_directory: str = "."
    ) -> None:
        dataset = read_json_model(dataset_path, RagDataset)
        self._retriever.load("data/processed")
        results = DatasetSearcher(retriever=self._retriever, k=k).search(
            dataset=dataset
        )
        write_json_model(
            f"{save_directory}/{Path(dataset_path).name}", results
        )

    def answer(self, query: str, k: int = 10) -> None:
        query = str(query)
        self._retriever.load("data/processed")
        sources = self._retriever.retrieve(query=query, k=k)
        sources_text = [read_source_text(source) for source in sources]
        generator = Generator()
        print(generator.generate(query, sources_text))

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str = "."
    ) -> None:
        search_results = read_json_model(
            student_search_results_path, StudentSearchResults
        )
        generator = Generator()
        results = DatasetAnswerer(generator).answer(search_results)
        write_json_model(
            f"{save_directory}/{Path(student_search_results_path).name}",
            results,
        )

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


if __name__ == "__main__":
    fire.Fire(CLI)
