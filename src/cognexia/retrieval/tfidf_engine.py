"""TF-IDF baseline retrieval engine for Cognexia."""

from __future__ import annotations

import math


class TfidfRetrievalEngine:
    """Sparse TF-IDF retrieval engine with cosine similarity ranking."""

    def __init__(self) -> None:
        self.vocabulary: set[str] = set()
        self.idf: dict[str, float] = {}
        self.doc_vectors: dict[str, dict[str, float]] = {}
        self.doc_norms: dict[str, float] = {}
        self.total_docs: int = 0

    def build_index(self, documents: dict[str, str]) -> None:
        """Build a TF-IDF index from preprocessed documents.

        Args:
            documents: Mapping of ``doc_id`` to preprocessed text.
        """

        self.vocabulary = set()
        self.idf = {}
        self.doc_vectors = {}
        self.doc_norms = {}
        self.total_docs = len(documents)

        term_counts_by_doc: dict[str, dict[str, int]] = {}
        total_terms_by_doc: dict[str, int] = {}
        document_frequency: dict[str, int] = {}

        for doc_id, text in documents.items():
            terms = text.split()
            total_terms_by_doc[doc_id] = len(terms)

            term_counts: dict[str, int] = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            term_counts_by_doc[doc_id] = term_counts

            for term in term_counts:
                document_frequency[term] = document_frequency.get(term, 0) + 1

        self.vocabulary = set(document_frequency.keys())

        for term, df in document_frequency.items():
            self.idf[term] = math.log((self.total_docs + 1) / (df + 1)) + 1

        for doc_id, term_counts in term_counts_by_doc.items():
            total_terms = total_terms_by_doc[doc_id]

            if total_terms == 0:
                self.doc_vectors[doc_id] = {}
                self.doc_norms[doc_id] = 0.0
                continue

            vector: dict[str, float] = {}
            for term, count in term_counts.items():
                tf = count / total_terms
                vector[term] = tf * self.idf[term]

            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = self._vector_norm(vector)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-k matching documents ranked by cosine similarity.

        Args:
            query: Preprocessed query text.
            top_k: Maximum number of results.

        Returns:
            Ranked ``(doc_id, score)`` tuples with strictly positive scores.
        """

        if top_k <= 0:
            return []

        query_terms = query.split()
        if not query_terms or not self.doc_vectors:
            return []

        query_term_counts: dict[str, int] = {}
        for term in query_terms:
            if term in self.idf:
                query_term_counts[term] = query_term_counts.get(term, 0) + 1

        total_query_terms = sum(query_term_counts.values())
        if total_query_terms == 0:
            return []

        query_vector: dict[str, float] = {}
        for term, count in query_term_counts.items():
            tf = count / total_query_terms
            query_vector[term] = tf * self.idf[term]

        query_norm = self._vector_norm(query_vector)
        if query_norm == 0.0:
            return []

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

    def _vector_norm(self, vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def _cosine_similarity(
        self,
        *,
        query_vector: dict[str, float],
        query_norm: float,
        doc_vector: dict[str, float],
        doc_norm: float,
    ) -> float:
        if query_norm == 0.0 or doc_norm == 0.0:
            return 0.0

        if len(query_vector) <= len(doc_vector):
            smaller = query_vector
            larger = doc_vector
        else:
            smaller = doc_vector
            larger = query_vector

        dot_product = 0.0
        for term, value in smaller.items():
            dot_product += value * larger.get(term, 0.0)

        denominator = query_norm * doc_norm
        if denominator == 0.0:
            return 0.0

        return dot_product / denominator