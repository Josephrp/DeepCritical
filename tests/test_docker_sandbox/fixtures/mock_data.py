"""
Mock data generators for Docker sandbox testing.
"""

import tempfile
from pathlib import Path


def create_test_file(content: str = "test content", filename: str = "test.txt") -> Path:
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=filename, delete=False) as f:
        f.write(content)
        return Path(f.name)


def create_test_directory() -> Path:
    """Create a temporary test directory."""
    return Path(tempfile.mkdtemp())


def create_nested_directory_structure() -> Path:
    """Create a nested directory structure for testing."""
    base_dir = Path(tempfile.mkdtemp())

    # Create nested structure
    (base_dir / "level1").mkdir()
    (base_dir / "level1" / "level2").mkdir()
    (base_dir / "level1" / "level2" / "level3").mkdir()

    # Add some files
    (base_dir / "level1" / "file1.txt").write_text("content1")
    (base_dir / "level1" / "level2" / "file2.txt").write_text("content2")
    (base_dir / "level1" / "level2" / "level3" / "file3.txt").write_text("content3")

    return base_dir
