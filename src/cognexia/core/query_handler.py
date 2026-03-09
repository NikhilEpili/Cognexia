"""Query orchestration for end-to-end retrieval pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from cognexia.ingestion.scanner import scan_directory
from cognexia.nlp.preprocessing import preprocess_text
from cognexia.retrieval.embedding_engine import EmbeddingRetrievalEngine
from cognexia.retrieval.tfidf_engine import TfidfRetrievalEngine


RetrievalMode = Literal["tfidf", "embedding"]
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".py"}


class QueryHandler:
    """Coordinate loading, preprocessing, indexing, and retrieval.

    The handler is stateful only for index caching. It builds the retrieval
    index once on first query, then reuses the same engine for subsequent
    queries.
    """

    def __init__(
        self,
        root_path: str | None = None,
        retrieval_mode: RetrievalMode | None = None,
        embedding_model: Any | None = None,
    ) -> None:
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.retrieval_mode = retrieval_mode or os.getenv(
            "COGNEXIA_RETRIEVAL_MODE", "tfidf"
        )
        self.engine = self._create_engine(self.retrieval_mode, embedding_model)
        self.index_built: bool = False

    def query(self, query_text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Run retrieval for a user query and return ranked results."""
        self._build_index_if_needed()

        if not query_text.strip() or top_k <= 0:
            return []

        processed_query = preprocess_text(
            query_text,
            remove_punctuation=True,
            remove_stopwords=True,
            lowercase=True,
        )
        if not processed_query:
            return []

        return self.engine.search(processed_query, top_k=top_k)

    def _build_index_if_needed(self) -> None:
        if self.index_built:
            return

        raw_documents = self._load_documents()
        processed_documents = self._preprocess_documents(raw_documents)
        self.engine.build_index(processed_documents)
        self.index_built = True

    def _load_documents(self) -> dict[str, str]:
        """Load UTF-8 text content from supported files under root path."""
        file_paths = scan_directory(str(self.root_path))
        documents: dict[str, str] = {}

        for file_path in sorted(file_paths):
            candidate = Path(file_path)
            if candidate.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
                continue

            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            documents[str(candidate)] = content

        return documents

    def _preprocess_documents(self, documents_raw: dict[str, str]) -> dict[str, str]:
        """Normalize raw documents before indexing."""
        processed: dict[str, str] = {}

        for doc_id, text in documents_raw.items():
            processed[doc_id] = preprocess_text(
                text,
                remove_punctuation=True,
                remove_stopwords=True,
                lowercase=True,
            )

        return processed

    def _create_engine(
        self,
        retrieval_mode: str,
        embedding_model: Any | None,
    ) -> TfidfRetrievalEngine | EmbeddingRetrievalEngine:
        if retrieval_mode == "tfidf":
            return TfidfRetrievalEngine()

        if retrieval_mode == "embedding":
            return EmbeddingRetrievalEngine(model=embedding_model)

        raise ValueError(
            "Invalid retrieval mode. Expected 'tfidf' or 'embedding'."
        )
