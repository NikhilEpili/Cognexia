import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.retrieval.chunker import chunk_document


def generate_words(count: int) -> str:
    """Generate deterministic space-separated fake words."""
    return " ".join(f"word{i}" for i in range(count))


class TestChunker(unittest.TestCase):
    def test_normal_case_2000_words(self):
        text = generate_words(2000)

        chunks = chunk_document(
            text=text,
            doc_id="docA",
            source_path="/tmp/docA.txt",
            chunk_size=800,
            overlap=100,
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]["text"].split()), 800)
        self.assertEqual(len(chunks[1]["text"].split()), 800)
        self.assertEqual(len(chunks[2]["text"].split()), 600)

        self.assertEqual(chunks[0]["text"].split()[0], "word0")
        self.assertEqual(chunks[1]["text"].split()[0], "word700")
        self.assertEqual(chunks[2]["text"].split()[0], "word1400")

    def test_small_document_less_than_chunk_size(self):
        text = generate_words(50)

        chunks = chunk_document(
            text=text,
            doc_id="docSmall",
            source_path="/tmp/small.txt",
            chunk_size=800,
            overlap=100,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], text)
        self.assertEqual(chunks[0]["chunk_id"], "docSmallchunk0")
        self.assertEqual(chunks[0]["position"], 0)

    def test_empty_text(self):
        chunks = chunk_document(
            text="",
            doc_id="docEmpty",
            source_path="/tmp/empty.txt",
            chunk_size=800,
            overlap=100,
        )

        self.assertEqual(chunks, [])

    def test_invalid_overlap(self):
        text = generate_words(20)

        with self.assertRaises(ValueError):
            chunk_document(
                text=text,
                doc_id="docInvalid",
                source_path="/tmp/x.txt",
                chunk_size=10,
                overlap=10,
            )

        with self.assertRaises(ValueError):
            chunk_document(
                text=text,
                doc_id="docInvalid",
                source_path="/tmp/x.txt",
                chunk_size=10,
                overlap=-1,
            )

        with self.assertRaises(ValueError):
            chunk_document(
                text=text,
                doc_id="docInvalid",
                source_path="/tmp/x.txt",
                chunk_size=0,
                overlap=0,
            )

    def test_metadata_integrity(self):
        text = generate_words(12)
        doc_id = "metaDoc"
        source_path = "C:/docs/meta.txt"

        chunks = chunk_document(
            text=text,
            doc_id=doc_id,
            source_path=source_path,
            chunk_size=5,
            overlap=2,
        )

        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk["doc_id"], doc_id)
            self.assertEqual(chunk["source_path"], source_path)
            self.assertEqual(chunk["position"], index)
            self.assertEqual(chunk["chunk_id"], f"{doc_id}chunk{index}")
            self.assertIsInstance(chunk["text"], str)
            self.assertEqual(chunk["text"], chunk["text"].strip())

    def test_correct_overlap_behavior(self):
        text = generate_words(12)

        chunks = chunk_document(
            text=text,
            doc_id="docOverlap",
            source_path="/tmp/overlap.txt",
            chunk_size=5,
            overlap=2,
        )

        self.assertEqual(len(chunks), 4)

        c0 = chunks[0]["text"].split()
        c1 = chunks[1]["text"].split()
        c2 = chunks[2]["text"].split()
        c3 = chunks[3]["text"].split()

        self.assertEqual(c0, ["word0", "word1", "word2", "word3", "word4"])
        self.assertEqual(c1, ["word3", "word4", "word5", "word6", "word7"])
        self.assertEqual(c2, ["word6", "word7", "word8", "word9", "word10"])
        self.assertEqual(c3, ["word9", "word10", "word11"])

        self.assertEqual(c0[-2:], c1[:2])
        self.assertEqual(c1[-2:], c2[:2])
        self.assertEqual(c2[-2:], c3[:2])


if __name__ == "__main__":
    unittest.main()
