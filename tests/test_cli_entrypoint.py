import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.cli.main import main

class TestCLIEntryPoint(unittest.TestCase):
    def test_query_command_runs(self):
        exit_code = main(["query", "hello world"])
        self.assertEqual(exit_code, 0)

    def test_empty_args_fails(self):
        exit_code = main([])
        self.assertNotEqual(exit_code, 0)

if __name__ == "__main__":
    unittest.main()