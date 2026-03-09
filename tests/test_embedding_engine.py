import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.retrieval.embedding_engine import EmbeddingRetrievalEngine


class FakeModel:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class TestEmbeddingEngine(unittest.TestCase):
    def test_index_builds_successfully(self):
        vectors = {
            "neural network": [1.0, 0.0],
            "python code": [0.0, 1.0],
        }
        engine = EmbeddingRetrievalEngine(model=FakeModel(vectors))

        engine.build_index(
            {
                "doc1": "neural network",
                "doc2": "python code",
            }
        )

        self.assertEqual(engine.total_docs, 2)
        self.assertEqual(set(engine.doc_vectors.keys()), {"doc1", "doc2"})
        self.assertGreater(engine.doc_norms["doc1"], 0.0)

    def test_query_returns_ranked_results(self):
        vectors = {
            "semantic search": [1.0, 0.0],
            "neural data": [0.8, 0.2],
            "cooking recipe": [0.1, 1.0],
        }
        engine = EmbeddingRetrievalEngine(model=FakeModel(vectors))
        engine.build_index(
            {
                "doc_a": "neural data",
                "doc_b": "cooking recipe",
            }
        )

        results = engine.search("semantic search", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "doc_a")
        self.assertEqual(results[1][0], "doc_b")
        self.assertGreater(results[0][1], results[1][1])

    def test_empty_document_dict_handled(self):
        engine = EmbeddingRetrievalEngine(model=FakeModel({}))
        engine.build_index({})

        self.assertEqual(engine.total_docs, 0)
        self.assertEqual(engine.doc_vectors, {})
        self.assertEqual(engine.search("anything", top_k=3), [])

    def test_search_before_build_index_raises_runtime_error(self):
        engine = EmbeddingRetrievalEngine(model=FakeModel({"q": [1.0]}))

        with self.assertRaises(RuntimeError):
            engine.search("q")

    def test_similar_query_ranks_expected_document_higher(self):
        vectors = {
            "semantic query": [1.0, 1.0],
            "semantic match": [1.0, 1.0],
            "different topic": [1.0, 0.1],
        }
        engine = EmbeddingRetrievalEngine(model=FakeModel(vectors))
        engine.build_index(
            {
                "doc_good": "semantic match",
                "doc_bad": "different topic",
            }
        )

        results = engine.search("semantic query", top_k=2)

        self.assertEqual(results[0][0], "doc_good")
        self.assertAlmostEqual(results[0][1], 1.0, places=9)
        self.assertEqual(results[1][0], "doc_bad")
        self.assertLess(results[1][1], 1.0)


if __name__ == "__main__":
    unittest.main()
