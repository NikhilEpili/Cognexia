import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cognexia.ingestion.scanner import scan_directory, scan_directory_with_metadata


class TestScanner(unittest.TestCase):
    def test_scan_empty_folder_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = scan_directory(temp_dir)
            self.assertEqual(results, [])

    def test_scan_supported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.txt").write_text("hello", encoding="utf-8")
            (root / "b.pdf").write_text("pdf", encoding="utf-8")
            (root / "c.md").write_text("md", encoding="utf-8")
            (root / "d.py").write_text("print('x')", encoding="utf-8")
            (root / "e.jpg").write_text("nope", encoding="utf-8")

            results = scan_directory(temp_dir)
            expected = {
                str(root / "a.txt"),
                str(root / "b.pdf"),
                str(root / "c.md"),
                str(root / "d.py"),
            }

            self.assertEqual(set(results), expected)

    def test_scan_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.bin").write_text("bin", encoding="utf-8")
            (root / "b.exe").write_text("exe", encoding="utf-8")

            results = scan_directory(temp_dir)
            self.assertEqual(results, [])

    def test_scan_nested_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested" / "deeper"
            nested.mkdir(parents=True)
            (nested / "note.md").write_text("nested", encoding="utf-8")

            results = scan_directory(temp_dir)
            self.assertIn(str(nested / "note.md"), results)

    def test_permission_error_logs_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("cognexia.ingestion.scanner.os.walk") as mocked_walk:
                def fake_walk(top, topdown=True, onerror=None, followlinks=False):
                    if onerror:
                        onerror(PermissionError("Denied"))
                    return []

                mocked_walk.side_effect = fake_walk
                with self.assertLogs("cognexia.ingestion.scanner", level="WARNING") as logs:
                    results = scan_directory(temp_dir)

            self.assertEqual(results, [])
            self.assertTrue(any("Permission denied" in message for message in logs.output))

    def test_scan_directory_with_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            file_path = root / "a.txt"
            file_path.write_text("hello", encoding="utf-8")

            results = scan_directory_with_metadata(temp_dir)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].path, str(file_path))
            self.assertIsNotNone(results[0].size_bytes)
            self.assertIsNotNone(results[0].last_modified)


if __name__ == "__main__":
    unittest.main()
