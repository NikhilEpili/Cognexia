"""Inference facade that delegates to the core query handler."""

from __future__ import annotations

from cognexia.core.query_handler import QueryHandler

_QUERY_HANDLER = QueryHandler()


def handle_query(query: str) -> list[tuple[str, float]]:
    """Stable query execution contract for the CLI layer.

    Returns:
        Ranked ``(doc_id, similarity_score)`` tuples.
    """

    return _QUERY_HANDLER.query(query)