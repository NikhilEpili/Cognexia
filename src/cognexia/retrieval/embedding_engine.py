"""Dense embedding retrieval engine for Cognexia."""

from __future__ import annotations

import math
from typing import Any


class EmbeddingRetrievalEngine:
    """Embedding-based retrieval with manual cosine similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self.model: Any | None = model
        self.doc_vectors: dict[str, list[float]] = {}
        self.doc_norms: dict[str, float] = {}
        self.total_docs: int = 0
        self._index_built = False

    def build_index(self, documents: dict[str, str]) -> None:
        """Build an embedding index from preprocessed document text.

        Args:
            documents: Mapping of ``doc_id`` to preprocessed text.
        """

        self.total_docs = len(documents)
        self.doc_vectors = {}
        self.doc_norms = {}

        if not documents:
            self._index_built = True
            return

        model = self._ensure_model()

        doc_ids_to_encode: list[str] = []
        texts_to_encode: list[str] = []

        for doc_id, text in documents.items():
            if not text.strip():
                self.doc_vectors[doc_id] = []
                self.doc_norms[doc_id] = 0.0
                continue
            doc_ids_to_encode.append(doc_id)
            texts_to_encode.append(text)

        if texts_to_encode:
            encoded = model.encode(texts_to_encode)
            for doc_id, vector in zip(doc_ids_to_encode, encoded):
                dense_vector = self._to_dense_list(vector)
                self.doc_vectors[doc_id] = dense_vector
                self.doc_norms[doc_id] = self._vector_norm(dense_vector)

        self._index_built = True

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Search indexed documents and return ranked results.

        Args:
            query: Preprocessed query text.
            top_k: Maximum number of results to return.

        Returns:
            Ranked list of ``(doc_id, similarity_score)``.

        Raises:
            RuntimeError: If called before ``build_index``.
        """

        if not self._index_built:
            raise RuntimeError("Index has not been built. Call build_index first.")

        if top_k <= 0 or not query.strip() or not self.doc_vectors:
            return []

        model = self._ensure_model()
        query_vector = self._to_dense_list(model.encode([query])[0])
        query_norm = self._vector_norm(query_vector)

        scored: list[tuple[str, float]] = []
        for doc_id, doc_vector in self.doc_vectors.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            score = self._cosine_similarity(
                query_vector=query_vector,
                query_norm=query_norm,
                doc_vector=doc_vector,
                doc_norm=doc_norm,
            )
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required for embedding retrieval."
            ) from error

        self.model = SentenceTransformer(self._model_name, device="cpu")
        return self.model

    def _to_dense_list(self, vector: Any) -> list[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    def _vector_norm(self, vector: list[float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    def _cosine_similarity(
        self,
        *,
        query_vector: list[float],
        query_norm: float,
        doc_vector: list[float],
        doc_norm: float,
    ) -> float:
        denominator = query_norm * doc_norm
        if denominator == 0.0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(query_vector, doc_vector))
        return dot_product / denominator