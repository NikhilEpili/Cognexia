import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.retrieval.tfidf_engine import TfidfRetrievalEngine


class TestTfidfEngine(unittest.TestCase):
    def test_single_document_retrieval(self):
        engine = TfidfRetrievalEngine()
        engine.build_index({"doc1.txt": "neural network deep learning"})

        results = engine.search("neural network", top_k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1.txt")
        self.assertGreater(results[0][1], 0.0)

    def test_multiple_documents_ranking_order(self):
        engine = TfidfRetrievalEngine()
        engine.build_index(
            {
                "doc1.txt": "neural network deep learning",
                "doc2.txt": "python programming language",
                "doc3.txt": "neural network basics",
            }
        )

        results = engine.search("neural network", top_k=3)

        self.assertEqual([doc_id for doc_id, _ in results], ["doc3.txt", "doc1.txt"])
        self.assertGreater(results[0][1], results[1][1])

    def test_query_with_unseen_term(self):
        engine = TfidfRetrievalEngine()
        engine.build_index(
            {
                "doc1.txt": "alpha beta",
                "doc2.txt": "gamma delta",
            }
        )

        results = engine.search("unknownterm", top_k=3)

        self.assertEqual(results, [])

    def test_empty_documents(self):
        engine = TfidfRetrievalEngine()
        engine.build_index(
            {
                "doc1.txt": "",
                "doc2.txt": "",
            }
        )

        self.assertEqual(engine.total_docs, 2)
        self.assertEqual(engine.vocabulary, set())
        self.assertEqual(engine.search("anything", top_k=3), [])

    def test_cosine_similarity_correctness(self):
        engine = TfidfRetrievalEngine()
        engine.build_index(
            {
                "doc_a": "alpha beta",
                "doc_b": "alpha alpha beta",
            }
        )

        results = engine.search("alpha beta", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "doc_a")
        self.assertEqual(results[1][0], "doc_b")
        self.assertAlmostEqual(results[0][1], 1.0, places=9)
        self.assertLess(results[1][1], 1.0)


if __name__ == "__main__":
    unittest.main()
