import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.nlp.preprocessing import preprocess_text


class TestPreprocessing(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(preprocess_text("HELLO"), "hello")

    def test_trim_whitespace(self):
        self.assertEqual(preprocess_text("  hello  "), "hello")

    def test_punctuation_removal(self):
        text = "Hello, world!"
        cleaned = preprocess_text(text, remove_punctuation=True)
        self.assertEqual(cleaned, "hello world")

    def test_empty_input(self):
        self.assertEqual(preprocess_text(""), "")

    def test_multiple_spaces(self):
        self.assertEqual(preprocess_text("a   b\n\n c"), "a b c")

    def test_unicode_preserved(self):
        self.assertEqual(preprocess_text("Café mañana"), "café mañana")

    def test_stopword_removal(self):
        text = "this is the test"
        cleaned = preprocess_text(text, remove_stopwords=True)
        self.assertEqual(cleaned, "this test")


if __name__ == "__main__":
    unittest.main()
