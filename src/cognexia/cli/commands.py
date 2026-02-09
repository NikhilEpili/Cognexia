from __future__ import annotations
from dataclasses import dataclass

from cognexia.inference.engine import handle_query

MOCK_VERSION_RESPONSE = "[Cognexia] Version command not initialized yet."
MOCK_HELP_HEADER = "[Cognexia] Help"

@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    message: str

def handle_version() -> str:
    """Handle the version command."""
    return MOCK_VERSION_RESPONSE

def handle_help(help_text: str) -> str:
    """Return CLI help text."""
    return f"{MOCK_HELP_HEADER}\n\n{help_text.strip()}"

def route_command(command: str | None, query: str | None, help_text: str) -> CommandResult:
    """Route the parsed command to a handler.
    Args:
        command: Parsed command name (e.g., query, help, version).
        query: Query string if applicable.
        help_text: Parser-generated help text.
    """
    if command in (None, "help"):
        exit_code = 1 if command is None else 0
        return CommandResult(exit_code=exit_code, message=handle_help(help_text))

    if command == "version":
        return CommandResult(exit_code=0, message=handle_version())

    if command == "query":
        if not query:
            return CommandResult(exit_code=1, message=handle_help(help_text))
        return CommandResult(exit_code=0, message=handle_query(query))

    return CommandResult(
        exit_code=1,
        message=f"[Cognexia] Unknown command: {command}",
    )
