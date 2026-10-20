This project has been created as part of the 42 curriculum by 'yaabdoul'.

# Description

This project is called Rag (retrieval-augmented generation), it's a solution which has multiple goals we can pick two very important generic goals:

- Enhance LLMs capacity to answer using information from an indexed corpus, including information newer than their training data
- use private data with local retrieval and generation; the models must first be downloaded or available in the local cache

## examples:

- A local llm may not know information newer than its training data, for example a newer version of vllm; rag can provide this information if it is present in the indexed corpus
- let's imagine that we have an enterprise that wants an ai assistant using sensitive internal data; local retrieval and generation let us use this data without sending the prompts and source text to an external llm api

# architecture:

In our implementation we have Chunker -> Indexer -> Retriever -> context compression -> Generator. retrieval uses lexical BM25 scoring, and context compression uses a local embedding model before generation. before i started writing code or implementing it, i tried to make the project extensible , maintainable , readable ... , that's why the chunkers and retrievers use abstract base classes, while the incremental indexer extends the indexer.

## Components roles:

### Chunker:

parse vllm documentation (.md files) and code (.py files), and create a list of Chunk objects (check `src/models.py`). the indexer reads the files before passing their content to the chunker. unsupported file extensions are skipped.

### Indexer && retriever:

the indexer reads and chunks the corpus, then builds and saves the BM25 index through the retriever. the retriever takes a query and k, and returns up to k chunks ranked by lexical BM25 score. chunks with a score of 0 are excluded, so fewer than k results may be returned.

### generator:

read the retrieved source text from the original files, compress the context using `sentence-transformers/all-MiniLM-L6-v2`, and build a prompt for the local `Qwen/Qwen3-0.6B` model. the prompt instructs the model to use only the provided context and say when the answer is missing; this instruction does not guarantee that every generated answer is correct.

## patterns used:

in our implementation we use a simple factory, the strategy and template method patterns, and an abstract retriever interface:

### simple factory and strategy patterns:

before chunking a file (.md or .py), `get_chunker` selects a `CodeChunker` or `MarkdownChunker` according to its extension. this is a simple factory selecting a chunking strategy, rather than an abstract factory creating families of related objects.

### template method pattern:

all chunkers share the `Chunker.chunk` method, which calls `_parse` and then `_enforce_max_size`. each child implements its own parsing strategy: python uses ast, while markdown uses headings. the parent splits oversized chunks to enforce max_chunk_size in characters. both children follow the same Chunker interface, which makes it easier to add other chunking strategies.

### interface:

all retrievers share the capabilities load , save , index , retrieve, defined as abstract methods in the `Retriever` class in `src/retrieval/base.py`. `BM25Retriever` implements this interface. another retrieval method, such as embedding search, could implement the same interface; embedding search is not currently implemented as a retriever.

# Instructions

this project exposes the following commands. run them from the project root with Python 3.10 or newer and uv installed. `uv run` manages the project dependencies.

the default corpus is `data/raw`, the default index directory is `data/processed`, max_chunk_size defaults to 2000 characters, and k defaults to 10. run index before search or answer. keep the original source files available and unchanged when generating answers, because their text is read again using the saved character ranges.

## index:

```bash
uv run python3 -m src index
```

## search:

```bash
uv run python3 -m src search --query "ur query" --k 10
```

## answer:

```bash
uv run python3 -m src answer --query "ur query" --k 10
```

## search_dataset:

```bash
uv run python3 -m src search_dataset --dataset_path "path/to/dataset.json" --k 10
```

## answer_dataset:

```bash
uv run python3 -m src answer_dataset \
    --student_search_results_path "path/to/search_results.json" \
    --save_directory "data/output/search_results_and_answer"
```

## incremental indexing:

```bash
uv run python3 -m src index --incremental
```

unchanged files reuse their chunks when a previous index and fingerprints are available. changed or deleted files update the chunks, and the BM25 index is rebuilt. changing max_chunk_size or missing index/fingerprint data triggers full chunking.

## evaluate:

```bash
uv run python3 -m src evaluate \
    --student_search_results_path "path/to/search_results.json" \
    --dataset_path "path/to/reference_dataset.json"
```

## local http api:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

`POST /search` and `POST /answer` accept a json body such as `{"query": "ur query", "k": 10}`. search returns the question and retrieved source locations, while answer returns an `answer` field.

single-query results default to `data/output/search_results/query.json` for search and `data/output/search_results_and_answer/query.json` for answer. dataset commands use the input filename under the corresponding output directory, grouped by the input folder unless `--save_directory` is supplied. dataset input uses a `rag_questions` list; answer_dataset consumes the json produced by search_dataset. replace the example dataset paths with real files.

# Resources:

- <https://huggingface.co/docs/transformers/main/main_classes/text_generation>
- <https://www.3blue1brown.com/>
- <https://chatgpt.com>
- <https://www.geeksforgeeks.org/nlp/what-is-bm25-best-matching-25-algorithm/>
- <https://fr.wikipedia.org/wiki/Okapi_BM25>

# TESTS and edge cases and tools:

to handle errors without an unhandled traceback in the cli, the entry point catches pipeline errors, other exceptions, and keyboard interrupts during imports and command execution. errors exit with code 1, while keyboard interrupts exit with code 130.

custom exceptions are defined in `src/utils/exceptions.py`. `src/utils/validator.py` validates max_chunk_size in the cli and k in the retriever:

```text
max_chunk_size must be an integer between 1 and 2000 characters, inclusive
k must be an integer between 1 and the number of indexed chunks, inclusive
```

these are manual test scenarios, not an automated test suite. retrieval tests require a valid index; a positive k is valid only when enough chunks are indexed. file mutation tests should use a copy of the corpus and a separate index directory via `--data_path` and `--index_path`.

## tests examples:

1. `uv run python3 -m src index --max_chunk_size 100000000000000000000000000000000000000000000000000000000000` — rejected because the maximum is 2000.
2. `uv run python3 -m src index --max_chunk_size a` — rejected because the value must be an integer.
3. `uv run python3 -m src index --max_chunk_size -1` — rejected because the minimum is 1.
4. `uv run python3 -m src index --max_chunk_size 1` — valid, but produces very small chunks.
5. `uv run python3 -m src search --query "vllm" --k 1000000000000000000` — rejected if k exceeds the indexed chunk count.
6. `uv run python3 -m src search --query 1 --k 10` — the cli converts the query to a string; a numeric query is not rejected just because it is numeric.
7. `uv run python3 -m src search --query "dskjfhsdkjfhs" --k 10` — may return no sources when no indexed tokens match.
8. `uv run python3 -m src search --query "dskjfhsdkjfhs" --k -` — test how Fire interprets a standalone `-` for k and how the validator handles the resulting value.
9. `uv run python3 -m src search --query -` — test how Fire interprets a standalone `-` for the query.
10. remove read permission from a copied corpus file, then index the copy with an account that cannot read it — full indexing logs and skips unreadable files; restore the permissions afterwards.
11. replace a copied .md file with binary content that is not valid utf-8, then index the copy — full indexing logs and skips undecodable files.
12. put invalid python syntax in a copied .py file, then index the copy — the indexer logs the syntax error and falls back to markdown chunking for that file.
13. remove read permission from a saved index file, then run search with an account that cannot read it — the cli should report an error; restore permissions afterwards.
14. press ctrl+c during imports or command execution — the cli handles the interrupt and exits with code 130 once its handlers are active.
15. corrupt a saved index file, then run search — the cli reports the loading error. the index includes BM25 files and `chunks.pkl`, not only json files.
16. `uv run python3 -m src search --query "vllm" --index_path "missing-index-directory"` — reports a missing index if that directory does not exist.
17. run search before index, when no saved index exists — reports that index must be run first.
18. add a file with no extension to a copied corpus and index it — the file is logged and skipped.
19. create an index with `--incremental`, rewrite a copied indexed file with exactly the same content, then run index with `--incremental` again — unchanged content should reuse its chunks.
20. change max_chunk_size to another value between 1 and 2000 when running `index --incremental` — triggers full chunking.
