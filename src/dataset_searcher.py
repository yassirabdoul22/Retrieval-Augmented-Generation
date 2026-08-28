from typing import List

from tqdm import tqdm

from src.models import MinimalSearchResults, RagDataset, StudentSearchResults
from src.retrieval.base import Retriever


class DatasetSearcher:

    def __init__(self, retriever: Retriever, k: int):
        self.retriever = retriever
        self.k = k

    def search(self, dataset: RagDataset) -> StudentSearchResults:
        results: List[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions):
            sources = self.retriever.retrieve(question.question, self.k)
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )
        return StudentSearchResults(search_results=results, k=self.k)
