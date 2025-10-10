"""
Cutadapt server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestCutadaptServer(BaseBioinformaticsToolTest):
    """Test Cutadapt server functionality."""

    @property
    def tool_name(self) -> str:
        return "cutadapt"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_files": ["path/to/reads_1.fq"],
            "output_files": ["path/to/trimmed_1.fq"],
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTQ files for testing."""
        reads_file = tmp_path / "sample_reads.fq"

        # Create mock FASTQ file
        reads_file.write_text(
            "@read1\n"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
            "+\n"
            "IIIIIIIIIIIIIII\n"
            "@read2\n"
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n"
            "+\n"
            "IIIIIIIIIIIIIII\n"
        )

        return {"input_files": [reads_file]}

    @pytest.mark.optional
    def test_cutadapt_trim(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Cutadapt trim functionality."""
        params = {
            "input_files": [str(sample_input_files["input_files"][0])],
            "output_files": [str(sample_output_dir / "trimmed.fq.gz")],
            "quality_cutoff": 20,
            "minimum_length": 20,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
