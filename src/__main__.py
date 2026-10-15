"""Entry point: ``uv run python -m src <command>``."""

import sys


def main() -> None:
    """Run the CLI, turning every error into a clean message."""
    try:
        import fire

        from src.cli import CLI
        from src.utils.exceptions import (
            ChunkingError,
            IndexingError,
            RetrievalError,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[Error] failed to start: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        fire.Fire(CLI)
    except (ChunkingError, RetrievalError, IndexingError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
