"""
Bowtie2 server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestBowtie2Server(BaseBioinformaticsToolTest):
    """Test Bowtie2 server functionality."""

    @property
    def tool_name(self) -> str:
        return "bowtie2"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "index": "path/to/index",
            "input_files": ["path/to/reads_1.fq"],
            "output_file": "path/to/output.sam",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTQ files for testing."""
        reads_file = tmp_path / "sample_reads.fq"

        # Create mock FASTQ file
        reads_file.write_text(
            "@read1\n"
            "ATCGATCGATCG\n"
            "+\n"
            "IIIIIIIIIIII\n"
            "@read2\n"
            "GCTAGCTAGCTA\n"
            "+\n"
            "IIIIIIIIIIII\n"
        )

        return {"input_files": [reads_file]}

    @pytest.mark.optional
    def test_bowtie2_align(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Bowtie2 align functionality."""
        params = {
            "index": "test_index",
            "input_files": [str(sample_input_files["input_files"][0])],
            "output_file": str(sample_output_dir / "aligned.sam"),
            "threads": 1,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.optional
    def test_bowtie2_build(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Bowtie2 build functionality."""
        params = {
            "reference": str(sample_input_files["input_files"][0]),
            "index_basename": str(sample_output_dir / "test_index"),
            "threads": 1,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.optional
    def test_bowtie2_inspect(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test Bowtie2 inspect functionality."""
        params = {
            "index": str(sample_output_dir / "test_index"),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
