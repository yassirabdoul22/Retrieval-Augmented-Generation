"""Entry point: ``uv run python -m src <command>``."""

import sys


def main() -> None:
    """Run the CLI, turning expected errors into clean messages."""
    try:
        import fire

        from src.cli import CLI
        from src.exceptions import ChunkingError, IndexingError, RetrievalError
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)

    try:
        fire.Fire(CLI)
    except (ChunkingError, RetrievalError, IndexingError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()