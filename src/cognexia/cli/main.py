from __future__ import annotations
import argparse
import sys
from typing import Sequence
from cognexia.cli.commands import route_command

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognexia",
        description="Cognexia command-line interface.",
    )

    subparsers = parser.add_subparsers(dest="command")

    query_parser = subparsers.add_parser(
        "query",
        help="Submit a query to Cognexia.",
        description="Submit a query to Cognexia.",
    )
    query_parser.add_argument(
        "query",
        nargs=argparse.REMAINDER,
        help="Query text to submit.",
    )

    subparsers.add_parser(
        "help",
        help="Show help for Cognexia CLI.",
        description="Show help for Cognexia CLI.",
    )

    subparsers.add_parser(
        "version",
        help="Show CLI version information.",
        description="Show CLI version information.",
    )

    return parser

def _normalize_args(argv: Sequence[str]) -> list[str]:
    if not argv:
        return []

    if argv[0].startswith("-"):
        return list(argv)

    if argv[0] in {"query", "help", "version"}:
        return list(argv)

    return ["query", *argv]

def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    normalized = _normalize_args(args)
    parsed = parser.parse_args(normalized)

    query = " ".join(getattr(parsed, "query", []) or []).strip()
    help_text = parser.format_help()

    result = route_command(getattr(parsed, "command", None), query or None, help_text)
    print(result.message)
    return result.exit_code

if __name__ == "__main__":
    raise SystemExit(main())
