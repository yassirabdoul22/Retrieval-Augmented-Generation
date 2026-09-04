from typing import Dict, List

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluater:
    def evaluate(
        self, student_resulls: StudentSearchResults, reference: RagDataset
    ) -> Dict[str, float]:
        truth_by_id = {
            q.question_id: q.sources
            for q in reference.rag_questions
            if isinstance(q, AnsweredQuestion)
        }
        per_k_scores: Dict[int, List[float]] = {k: [] for k in (1, 3, 5, 10)}
        for r in student_resulls.search_results:
            true_sources = truth_by_id.get(r.question_id)
            if true_sources is None:
                continue
            for k in per_k_scores:
                per_k_scores[k].append(
                    self._question_recall(true_sources, r.retrieved_sources, k)
                )
        return {
            f"recall@{k}": (sum(scores) / len(scores)) if scores else 0.0
            for k, scores in per_k_scores.items()
        }

    def _compute_overlap_ratio(
        self, a: MinimalSource, b: MinimalSource
    ) -> float:
        if a.file_path != b.file_path:
            return 0.0
        intersection = max(
            0,
            min(a.last_character_index, b.last_character_index)
            - max(a.first_character_index, b.first_character_index),
        )
        union = max(a.last_character_index, b.last_character_index) - min(
            a.first_character_index, b.first_character_index
        )
        if union == 0:
            return 0.0
        return intersection / union

    def _is_source_matched(
        self,
        real_source: MinimalSource,
        retrieved: List[MinimalSource],
        threshold: float = 0.05,
    ) -> bool:
        return any(
            self._compute_overlap_ratio(real_source, r) >= threshold
            for r in retrieved
        )

    def _question_recall(
        self,
        real_sources: List[MinimalSource],
        retrived: List[MinimalSource],
        k: int,
    ) -> float:
        if not real_sources:
            return 0.0
        top_k = retrived[:k]
        found = sum(
            1 for t in real_sources if self._is_source_matched(t, top_k)
        )
        return found / len(real_sources)
