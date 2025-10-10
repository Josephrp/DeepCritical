"""
Deeptools server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestDeeptoolsServer(BaseBioinformaticsToolTest):
    """Test Deeptools server functionality."""

    @property
    def tool_name(self) -> str:
        return "deeptools"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "bam_file": "path/to/sample.bam",
            "output_file": "path/to/coverage.bw",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample BAM files for testing."""
        bam_file = tmp_path / "sample.bam"

        # Create mock BAM file (just a placeholder)
        bam_file.write_text("BAM file content")

        return {"bam_file": bam_file}

    @pytest.mark.optional
    def test_deeptools_bam_coverage(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test Deeptools bamCoverage functionality."""
        params = {
            "bam_file": str(sample_input_files["bam_file"]),
            "output_file": str(sample_output_dir / "coverage.bw"),
            "bin_size": 50,
            "normalize_using": "RPGC",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.optional
    def test_deeptools_compute_matrix(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test Deeptools computeMatrix functionality."""
        params = {
            "regions_file": str(sample_input_files["bam_file"]),
            "score_files": [str(sample_input_files["bam_file"])],
            "output_file": str(sample_output_dir / "matrix.mat.gz"),
            "reference_point": "TSS",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
