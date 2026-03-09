import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.core.query_handler import QueryHandler


class _FakeEmbeddingEngine:
    def __init__(self, model=None) -> None:
        _ = model
        self.build_calls = 0
        self.index = {}

    def build_index(self, documents: dict[str, str]) -> None:
        self.build_calls += 1
        self.index = documents

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        query_terms = set(query.split())
        for doc_id, text in self.index.items():
            overlap = len(query_terms.intersection(set(text.split())))
            if overlap > 0:
                scored.append((doc_id, float(overlap)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline_returns_ranked_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("neural network semantic", encoding="utf-8")
            (root / "b.txt").write_text("gardening tools plants", encoding="utf-8")

            handler = QueryHandler(root_path=str(root), retrieval_mode="tfidf")
            results = handler.query("semantic neural", top_k=3)

            self.assertGreater(len(results), 0)
            self.assertEqual(Path(results[0][0]).name, "a.txt")

    def test_empty_document_directory_handled_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = QueryHandler(root_path=temp_dir, retrieval_mode="tfidf")
            results = handler.query("anything", top_k=3)
            self.assertEqual(results, [])

    def test_switching_retrieval_modes_works(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "doc.txt").write_text("semantic search baseline", encoding="utf-8")

            tfidf_handler = QueryHandler(root_path=str(root), retrieval_mode="tfidf")
            tfidf_results = tfidf_handler.query("semantic", top_k=3)
            self.assertGreater(len(tfidf_results), 0)

            with mock.patch(
                "cognexia.core.query_handler.EmbeddingRetrievalEngine",
                _FakeEmbeddingEngine,
            ):
                embedding_handler = QueryHandler(
                    root_path=str(root),
                    retrieval_mode="embedding",
                )
                embedding_results = embedding_handler.query("semantic", top_k=3)
                self.assertGreater(len(embedding_results), 0)

    def test_index_builds_once_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "doc.txt").write_text("neural neural search", encoding="utf-8")

            with mock.patch(
                "cognexia.core.query_handler.TfidfRetrievalEngine"
            ) as engine_cls:
                engine_instance = mock.MagicMock()
                engine_instance.search.return_value = [(str(root / "doc.txt"), 1.0)]
                engine_cls.return_value = engine_instance

                handler = QueryHandler(root_path=str(root), retrieval_mode="tfidf")
                handler.query("neural")
                handler.query("search")

                self.assertEqual(engine_instance.build_index.call_count, 1)
                self.assertEqual(engine_instance.search.call_count, 2)

    def test_no_crash_on_repeated_queries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "x.txt").write_text("alpha beta gamma", encoding="utf-8")

            handler = QueryHandler(root_path=str(root), retrieval_mode="tfidf")
            first = handler.query("alpha")
            second = handler.query("beta")

            self.assertIsInstance(first, list)
            self.assertIsInstance(second, list)


if __name__ == "__main__":
    unittest.main()
