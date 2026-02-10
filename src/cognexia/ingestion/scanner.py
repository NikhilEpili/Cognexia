"""Local file system scanner for Cognexia."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = [".txt", ".pdf", ".md", ".py"]


@dataclass(frozen=True)
class FileMetadata:
    """Metadata for a discovered file."""

    path: str
    size_bytes: int | None
    last_modified: datetime | None


def _normalize_extensions(supported_extensions: List[str] | None) -> List[str]:
    if not supported_extensions:
        return DEFAULT_EXTENSIONS.copy()

    normalized: list[str] = []
    for ext in supported_extensions:
        if not ext:
            continue
        clean_ext = ext.lower().strip()
        if not clean_ext.startswith("."):
            clean_ext = f".{clean_ext}"
        normalized.append(clean_ext)

    return normalized


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def _handle_walk_error(error: OSError) -> None:
    if isinstance(error, PermissionError):
        logger.warning("Permission denied accessing %s", error.filename or "<unknown>")
        return

    logger.warning("Error accessing %s: %s", error.filename or "<unknown>", error)


def _iter_files(root: Path, ignore_hidden: bool) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=_handle_walk_error,
        followlinks=False,
    ):
        current_dir = Path(dirpath)
        if ignore_hidden:
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_hidden(current_dir / name)
            ]

        for filename in filenames:
            candidate = current_dir / filename
            if ignore_hidden and _is_hidden(candidate):
                continue
            yield candidate


def scan_directory(
    path: str,
    supported_extensions: List[str] | None = None,
    *,
    ignore_hidden: bool = True,
) -> List[str]:
    """Recursively scan the given directory and return matching file paths.

    Args:
        path: Path to the root directory or file.
        supported_extensions: List of allowed extensions.
            Defaults to ['.txt', '.pdf', '.md', '.py'].
        ignore_hidden: If True, skip dotfiles and dot-directories.

    Returns:
        List of full file paths as strings.
    """

    root = Path(path).expanduser()
    extensions = set(_normalize_extensions(supported_extensions))

    if not root.exists():
        logger.warning("Path does not exist: %s", root)
        return []

    if root.is_file():
        if root.suffix.lower() in extensions:
            return [str(root)]
        return []

    results: list[str] = []
    for file_path in _iter_files(root, ignore_hidden=ignore_hidden):
        if file_path.suffix.lower() in extensions:
            results.append(str(file_path))

    return results


def scan_directory_with_metadata(
    path: str,
    supported_extensions: List[str] | None = None,
    *,
    ignore_hidden: bool = True,
) -> List[FileMetadata]:
    """Recursively scan and return matching files with metadata.

    Args:
        path: Path to the root directory or file.
        supported_extensions: List of allowed extensions.
        ignore_hidden: If True, skip dotfiles and dot-directories.

    Returns:
        List of FileMetadata entries.
    """

    root = Path(path).expanduser()
    extensions = set(_normalize_extensions(supported_extensions))

    if not root.exists():
        logger.warning("Path does not exist: %s", root)
        return []

    if root.is_file():
        if root.suffix.lower() not in extensions:
            return []
        return [_build_metadata(root)]

    results: list[FileMetadata] = []
    for file_path in _iter_files(root, ignore_hidden=ignore_hidden):
        if file_path.suffix.lower() not in extensions:
            continue
        results.append(_build_metadata(file_path))

    return results


def _build_metadata(path: Path) -> FileMetadata:
    try:
        stat_info = path.stat()
        size = stat_info.st_size
        last_modified = datetime.fromtimestamp(stat_info.st_mtime)
    except OSError as error:
        if isinstance(error, PermissionError):
            logger.warning("Permission denied reading %s", path)
        else:
            logger.warning("Unable to stat %s: %s", path, error)
        size = None
        last_modified = None

    return FileMetadata(
        path=str(path),
        size_bytes=size,
        last_modified=last_modified,
    )
