"""
SAMtools server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)
from tests.utils.mocks.mock_data import create_mock_sam


class TestSAMtoolsServer(BaseBioinformaticsToolTest):
    """Test SAMtools server functionality."""

    @property
    def tool_name(self) -> str:
        return "samtools"

    @property
    def tool_class(self):
        # This would import the actual SAMtools server class
        # For now, we'll use a mock implementation
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {"input_file": "path/to/input.sam", "output_format": "bam"}

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample SAM file for testing."""
        sam_file = tmp_path / "sample.sam"
        create_mock_sam(sam_file, num_alignments=50)
        return {"input_file": sam_file}

    def test_sam_to_bam_conversion(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test SAM to BAM conversion functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "output_format": "bam",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

        # Verify BAM file was created
        bam_file = sample_output_dir / "sample.bam"
        assert bam_file.exists()

    def test_bam_indexing(self, tool_instance, sample_output_dir):
        """Test BAM indexing functionality."""
        # Create a mock BAM file first
        bam_file = sample_output_dir / "test.bam"
        bam_file.write_bytes(b"mock bam content")

        params = {
            "input_file": str(bam_file),
            "create_index": True,
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True

        # Verify index file was created
        bai_file = sample_output_dir / "test.bam.bai"
        assert bai_file.exists()

    def test_samtools_view(self, tool_instance, sample_input_files, sample_output_dir):
        """Test samtools view functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "output_format": "bam",
            "region": "chr1:1-1000",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
