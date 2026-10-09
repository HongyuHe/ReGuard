#!/usr/bin/env python3
"""Create a zip archive of the current project from the repository root."""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIX = f"{REPO_ROOT.name}_project_archive"
EXCLUDED_DIRECTORY_NAMES = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".hypothesis",
    ".eggs",
    "build",
    "dist",
}
EXCLUDED_FILE_NAMES = {
    ".DS_Store",
}
EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".partial",
}


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def _ensure_running_from_repo_root() -> None:
    current_working_directory = Path.cwd().resolve()
    if current_working_directory != REPO_ROOT:
        _fail(
            "Run this script from the project root directory:\n"
            f"  expected: {REPO_ROOT}\n"
            f"  current:  {current_working_directory}"
        )


def _is_generated_archive(path: Path) -> bool:
    return path.parent == REPO_ROOT and path.suffix == ".zip" and path.stem.startswith(
        ARCHIVE_PREFIX
    )


def _should_exclude_directory(directory_path: Path) -> bool:
    return directory_path.name in EXCLUDED_DIRECTORY_NAMES or directory_path.name.endswith(
        ".egg-info"
    )


def _should_exclude_file(file_path: Path) -> bool:
    if file_path.name in EXCLUDED_FILE_NAMES:
        return True
    if file_path.suffix in EXCLUDED_FILE_SUFFIXES:
        return True
    if file_path.name.endswith(".egg-info"):
        return True
    if _is_generated_archive(file_path):
        return True
    return False


def _iter_project_paths(project_root: Path) -> Iterator[Path]:
    for current_root, directory_names, file_names in os.walk(project_root, topdown=True):
        current_root_path = Path(current_root)
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if not _should_exclude_directory(current_root_path / directory_name)
        )

        for directory_name in directory_names:
            yield current_root_path / directory_name

        for file_name in sorted(file_names):
            file_path = current_root_path / file_name
            if _should_exclude_file(file_path):
                continue
            yield file_path


def _next_archive_path(project_root: Path) -> Path:
    archive_path = project_root / f"{ARCHIVE_PREFIX}.zip"
    archive_index = 1
    while archive_path.exists():
        archive_path = project_root / f"{ARCHIVE_PREFIX}_{archive_index:03d}.zip"
        archive_index += 1
    return archive_path


def _relative_archive_name(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def create_project_archive() -> Path:
    archive_path = _next_archive_path(REPO_ROOT)

    #* Skip prior archives from this script so repeated runs do not recurse.
    project_paths = list(_iter_project_paths(REPO_ROOT))

    temporary_archive = tempfile.NamedTemporaryFile(
        dir=REPO_ROOT,
        prefix=f"{ARCHIVE_PREFIX}_",
        suffix=".zip.partial",
        delete=False,
    )
    temporary_archive_path = Path(temporary_archive.name)
    temporary_archive.close()

    try:
        with zipfile.ZipFile(
            temporary_archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive_handle:
            for project_path in project_paths:
                archive_handle.write(project_path, arcname=_relative_archive_name(project_path))

        temporary_archive_path.replace(archive_path)
    except Exception:
        temporary_archive_path.unlink(missing_ok=True)
        raise

    return archive_path


def main() -> int:
    _ensure_running_from_repo_root()
    archive_path = create_project_archive()
    print(f"Created archive: {archive_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
