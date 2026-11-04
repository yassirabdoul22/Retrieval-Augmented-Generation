# Final validation after typing and README fixes

The actual Ubuntu moulinette supplied in the repository was executed through the unmodified exam scripts. Tests used an isolated copy of the updated application and the supplied private datasets. Existing original indexes and answer outputs were not overwritten.

## Changes

- `src/generator.py`: define a narrow typed generation protocol and cast the dynamically loaded causal language model to it. This removes the Transformers self-type mismatch without ignoring mypy diagnostics, changing model selection, or altering generation behavior.
- `README.md`: correct the opening attribution; document chunking, BM25 mechanics, design tradeoffs, CPU-only setup, commands, performance measurements, correction procedures, limitations, and the AI assistance used during this review.
- Other pre-existing working-tree changes were left in place.

## Executed checks

| Check | Result |
|---|---|
| Root `make lint` | PASS: flake8 and mandatory mypy flags, 27 source files |
| Supplied retrieval exam | PASS: 4/4 |
| Official private docs recall@5 | 0.800, threshold 0.800 |
| Official private code recall@5 | 0.600, threshold 0.500 |
| Official full indexing | 4 seconds, limit 300 seconds |
| Official 200-question throughput | PASS against 90-second limit; shell timer rounded the two subsecond batches to 0 seconds |
| Official student-output validation | Both private output datasets valid; all 100 questions evaluated per dataset |
| Moulinette list_valid_questions | Selected three questions marked VALID |
| Supplied answer exam | Completed, exit 0; three answers judged satisfactory on inspection |
| Supplied edge-case exam | PASS: 4/4 |

The answer exam took 81.026 seconds including three separate model loads; the edge-case exam took 1.218 seconds. Answers identify `vllm:num_requests_running`, SkyPilot, and the `generate_cmake_presets.py` helper with its configuration detection, matching the reference facts.

The supplied answer exam requires human judgment. Successful script execution is not an automated quality grade; 3/3 is this review's assessment. The oral student-understanding check remains outside automated validation. Performance depends on the machine and caches; docs recall is exactly at the required threshold.

## Evidence

- [lint.log](/tmp/rag-official-ovxkfc6o/lint.log)
- [retrieval-exam.log](/tmp/rag-official-ovxkfc6o/retrieval-exam.log)
- [evaluations/retrieval/2026-10-28_21-55-24/summary.log](/tmp/rag-official-ovxkfc6o/evaluations/retrieval/2026-10-28_21-55-24/summary.log)
- [evaluations/retrieval/2026-10-28_21-55-24/docs_eval.log](/tmp/rag-official-ovxkfc6o/evaluations/retrieval/2026-10-28_21-55-24/docs_eval.log)
- [evaluations/retrieval/2026-10-28_21-55-24/code_eval.log](/tmp/rag-official-ovxkfc6o/evaluations/retrieval/2026-10-28_21-55-24/code_eval.log)
- [answer-exam.log](/tmp/rag-official-ovxkfc6o/answer-exam.log)
- [evaluations/answer/2026-10-28_21-55-50/answer_1.log](/tmp/rag-official-ovxkfc6o/evaluations/answer/2026-10-28_21-55-50/answer_1.log) (and 2, 3)
- [edge-exam.log](/tmp/rag-official-ovxkfc6o/edge-exam.log)
- [evaluations/edge_cases/2026-10-28_21-57-11/summary.log](/tmp/rag-official-ovxkfc6o/evaluations/edge_cases/2026-10-28_21-57-11/summary.log)

No application code imports or invokes the moulinette; only the external correction scripts do so. No fixes to retrieval, generation logic, datasets, or exam scripts were needed.

Evidence logs are stored in the temporary validation directory linked above; they may be removed when temporary files are cleaned.
