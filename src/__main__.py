"""Entry point: ``uv run python -m src <command>``."""

import sys


def main() -> None:
    """Run the CLI, turning every error into a clean message."""
    try:
        import fire

        from src.cli import CLI
        from src.utils.exceptions import PipelineError
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return
    except Exception as e:
        print(f"[Error] failed to start: {e}", file=sys.stderr)
        return

    try:
        fire.Fire(CLI)
    except PipelineError as e:
        print(e, file=sys.stderr)
        return
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
