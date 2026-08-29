from typing import List, cast

from tqdm import tqdm

from src.generator import Generator
from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.source_loader import read_source_text


class DatasetAnswerer:

    def __init__(self, generator: Generator):
        self.generator = generator

    def answer(
        self, search_results: StudentSearchResults
    ) -> StudentSearchResultsAndAnswer:
        results: List[MinimalAnswer] = []
        for result in tqdm(search_results.search_results):
            sources_text = [
                read_source_text(source)
                for source in result.retrieved_sources
            ]
            answer_text = self.generator.generate(
                result.question, sources_text
            )
            results.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=cast(
                        List[MinimalSource], result.retrieved_sources
                    ),
                    answer=answer_text,
                )
            )
        return StudentSearchResultsAndAnswer(
            search_results=results, k=search_results.k
        )
