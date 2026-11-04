*This project has been created as part of the 42 curriculum by yaabdoul.*

# RAG against the machine

## Description

A local retrieval-augmented generation (RAG) system for answering questions about
the supplied vLLM codebase. It indexes Python source and Markdown documentation,
retrieves relevant passages with BM25, and passes them to `Qwen/Qwen3-0.6B` to
generate an answer.

RAG supplies information from a chosen corpus without retraining the language
model. It can provide information newer than the model's training data and let a
local assistant use private documents. Dependencies and model weights must first
be downloaded or available in the local cache; prompts and source text are not
sent to an external generation API.

## Architecture

```text
Python / Markdown files
        -> chunking -> persisted BM25 index
Question -> BM25 retrieval -> original source passages -> local Qwen answer
```

- **Chunkers** produce `Chunk` objects containing text, a file path, and an
  end-exclusive character range. Unsupported file types are skipped.
- **Indexer** reads the corpus, selects a chunker, builds the BM25 index, and
  saves it with its chunks. Unreadable files are logged and skipped; Python
  syntax errors fall back to Markdown chunking.
- **Retriever** ranks chunks and returns up to `k` sources with positive scores.
- **Generator** rereads those source ranges and builds a chat prompt. It uses a
  4096-token context limit, reserves 96 tokens for the answer, disables thinking,
  and uses greedy generation. Sources keep their retrieval order; the last
  retained source can be shortened to fit. There is no semantic compression or
  reranking.
- **RagService** shares the single-question flow between the CLI and HTTP API.
  It loads the index and generation model on first use.
- **Dataset helpers** preserve question IDs while processing batches. Pydantic
  models validate the JSON exchanged between stages.

### Chunking strategy

`CodeChunker` parses Python with `ast` and splits at the ends of top-level
functions, asynchronous functions, and classes, preserving intervening text.
`MarkdownChunker` splits at headings and retains any introductory text.

Both strategies then split oversized sections into consecutive pieces of at
most `max_chunk_size` characters, with no overlap. The default and maximum are
2000 characters. Smaller chunks give more precise locations but may separate a
fact from its explanation. Larger chunks preserve more context but introduce
noise and use more of the generator's token budget. Oversized classes can still
be cut inside methods by the final size-enforcement step.

### Retrieval method

`BM25Retriever` uses `bm25s`. Each indexed document combines its file path, with
slashes replaced by spaces, and the chunk text. This lets file names contribute
to matching alongside document content.

BM25 gives more weight to rare matching terms, saturates the contribution of
repeated terms, and normalizes for document length. Queries use the same
library's tokenization. Results retain descending BM25 rank; zero-score chunks
are excluded, so a search can return fewer than `k` sources. Lexical search is
fast and useful for exact identifiers, but paraphrases with little vocabulary
overlap can be missed.

### Design decisions and challenges

- A simple factory chooses the chunking strategy by extension. The shared
  `Chunker.chunk` method is a template method: parse first, then enforce size.
  Abstract chunker and retriever interfaces make additional strategies possible.
- BM25 provides a small, CPU-friendly baseline without an embedding model or
  vector index. Semantic and hybrid retrieval are not implemented.
- Character ranges match the grading format and avoid duplicating source text
  in result JSON. Original files must remain available and consistent with the
  index: generation rereads them, so changed files require reindexing.
- Fitting a small model's context window requires counting the complete rendered
  prompt, including instructions and the question. The generator rejects a
  question that cannot fit even without sources and trims context by token
  count. Prioritizing retrieval order can omit useful lower-ranked evidence.
- The prompt asks for one or two grounded sentences and an explicit unknown
  response when evidence is missing. This improves grounding but cannot
  guarantee factual correctness; the small model can still produce incomplete
  or incorrect answers.
- A typed `CausalLanguageModel` protocol describes the tensor-generation method
  the application uses. The dynamically loaded Transformers model is cast to
  that interface without depending on Transformers' private typing helpers.

## Instructions

Run commands from the repository root with Python 3.10+ and `uv` installed.
Place the supplied vLLM repository under `data/raw/vllm-0.10.1/`.

```bash
uv sync
make lint
uv run python -m src index
uv run python -m src search "How to configure OpenAI server?" --k 10
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

PyTorch is configured to use its CPU-only package index, avoiding CUDA package
downloads. The first answer may download the Qwen model and tokenizer. Cached
weights support subsequent local runs.

### Paths and options

Defaults are `data/raw` for the corpus, `data/processed` for the index,
`max_chunk_size=2000`, and `k=10`. Index before searching or answering. A valid
`k` is an integer between 1 and the number of indexed chunks.

```bash
uv run python -m src index --data_path data/raw \
    --index_path data/processed --max_chunk_size 2000
uv run python -m src search "PagedAttention" --k 5 \
    --index_path data/processed --output_path data/output/paged_attention.json
```

Single searches default to `data/output/search_results/query.json`; single
answers default to `data/output/search_results_and_answer/query.json`. Both
commands accept `--output_path` and `--index_path`.

### Dataset processing and evaluation

Dataset JSON contains a `rag_questions` list of questions, optionally with
reference answers and sources. Replace the example paths with your dataset
locations. `answer_dataset` consumes the output of `search_dataset`.

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json
```

Batch output keeps the input filename. Without `--save_directory`, results go
under the corresponding output root, grouped by the input folder name.

`evaluate` reports macro-averaged recall@1, @3, @5, and @10. A reference source
matches when a retrieved range has the same file path and character-range
intersection-over-union of at least 0.05. The official correction uses the
provided moulinette separately; the application never imports it.

### Incremental indexing and caching

```bash
uv run python -m src index --incremental
```

File fingerprints use modification time, size, and, when metadata changes, a
content hash. Unchanged files reuse their chunks; changed and deleted files
update the chunk set. BM25 is still fully rebuilt. Missing index/fingerprint
data or a changed chunk size triggers full chunking.

The HTTP service retains its loaded index and model between requests. There is
no query-result cache, and separate CLI processes load their own state.

### Local HTTP API

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"query":"What is PagedAttention?","k":10}'
```

`POST /search` returns a question and ranked source locations. `POST /answer`
accepts the same request and returns an `answer` field. Empty queries and invalid
result counts receive HTTP errors.

## Performance and validation

Measured on 2026-10-28 with the supplied vLLM corpus, 2000-character chunks,
`k=10`, Python 3.13.1, CPU-only PyTorch 2.14.1, and Transformers 5.21.0. The
Ubuntu moulinette validated fresh outputs for both 100-question private datasets:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required recall@5 |
|---|---:|---:|---:|---:|---:|
| Docs | 56% | 79% | 80% | 89% | 80% |
| Code | 34% | 56% | 60% | 68% | 50% |

The supplied retrieval exam passed all four checks. Its indexing run took
4 seconds, within the 300-second limit, and its 200-question retrieval run
passed the 90-second limit. The script uses whole-second timestamps; a separate
fine-grained validation measured approximately 4.7 seconds for indexing and
1.4 seconds for the two search batches. These measurements use cached
dependencies and do not include initial package or model downloads. Runtime
varies by machine and filesystem cache. Docs recall meets the threshold exactly,
so changes to chunking or tokenization require careful reevaluation.

The supplied answer exam produced three satisfactory answers about the
running-requests metric, SkyPilot, and generating CMake presets. The moulinette
confirmed that their reference sources were retrieved. Answer quality remains a
manual judgment, not a guarantee for every question. The three separate CLI
answer runs took about 81 seconds together, including model loading. An earlier
batch run took about 24 seconds per answer after loading the model once. No
chunk-size sweep has been measured; the table describes only the default size.

## Tests and error handling

```bash
make lint
bash exams/scripts/exam_edge_cases.sh --student-path .
```

The supplied edge-case exam checks empty and gibberish queries, `k=0`, and a
missing dataset. All four passed during validation. The CLI catches pipeline
errors and unexpected exceptions, prints an error, and exits with code 1;
keyboard interrupts exit with code 130.

Additional isolated checks covered Unicode chunk offsets, source-size limits,
incremental updates and deletions, index reuse, HTTP search and answer requests,
and the generation token budget. These validation checks are not a committed
unit-test suite. Use copied files and a separate `--data_path` / `--index_path`
when testing corrupt files, invalid Python, or permissions.

For correction, arrange `student/`, `exams/`, the moulinette binary, and
`data/datasets/private/` under one evaluation root, as described in the supplied
exam guide. Run the official retrieval script from there:

```bash
bash exams/scripts/exam_retrieval.sh \
    --student-path ./student --moulinette-path ./moulinette/moulinette-ubuntu
```

Private datasets and the moulinette belong to the correction setup and are not
application dependencies. The answer exam requires manual judgment, and the
oral understanding questions must still be answered by the student.

## Resources and AI assistance

- [Transformers text generation](https://huggingface.co/docs/transformers/main/main_classes/text_generation)
- [bm25s implementation](https://github.com/xhluca/bm25s)
- [Okapi BM25 background](https://en.wikipedia.org/wiki/Okapi_BM25)
- [ChatGPT](https://chatgpt.com)

AI assistance was used in this review to inspect the pipeline, diagnose the
Transformers/mypy typing mismatch, improve this README, and run and interpret
validation checks. Reported measurements come from executed commands. The
student remains responsible for understanding the implementation and explaining
its decisions during the defense.
