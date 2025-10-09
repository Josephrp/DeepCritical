"""
BEDTools server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestBEDToolsServer(BaseBioinformaticsToolTest):
    """Test BEDTools server functionality."""

    @property
    def tool_name(self) -> str:
        return "bedtools"

    @property
    def tool_class(self):
        # This would import the actual BEDTools server class
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_file_a": "path/to/file_a.bed",
            "input_file_b": "path/to/file_b.bed",
            "operation": "intersect",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample BED files for testing."""
        bed_a = tmp_path / "regions_a.bed"
        bed_b = tmp_path / "regions_b.bed"

        # Create mock BED files
        bed_a.write_text("chr1\t100\t200\tfeature1\nchr1\t300\t400\tfeature2\n")
        bed_b.write_text("chr1\t150\t250\tpeak1\nchr1\t350\t450\tpeak2\n")

        return {"input_file_a": bed_a, "input_file_b": bed_b}

    def test_bedtools_intersect(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test BEDTools intersect functionality."""
        params = {
            "input_file_a": str(sample_input_files["input_file_a"]),
            "input_file_b": str(sample_input_files["input_file_b"]),
            "operation": "intersect",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

        # Verify output file was created
        output_file = sample_output_dir / "intersect_output.bed"
        assert output_file.exists()

        # Verify output content
        content = output_file.read_text()
        assert "chr1" in content

    def test_bedtools_merge(self, tool_instance, sample_input_files, sample_output_dir):
        """Test BEDTools merge functionality."""
        params = {
            "input_file_a": str(sample_input_files["input_file_a"]),
            "operation": "merge",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    def test_bedtools_coverage(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test BEDTools coverage functionality."""
        params = {
            "input_file_a": str(sample_input_files["input_file_a"]),
            "input_file_b": str(sample_input_files["input_file_b"]),
            "operation": "coverage",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
