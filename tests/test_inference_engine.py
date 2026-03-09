import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.inference.engine import handle_query


class TestInferenceEngine(unittest.TestCase):
    def test_handle_query_returns_ranked_results(self):
        response = handle_query("hello")
        self.assertIsInstance(response, list)
        for item in response:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


if __name__ == "__main__":
    unittest.main()
