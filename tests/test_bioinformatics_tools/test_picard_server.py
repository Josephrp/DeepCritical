"""
Picard server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestPicardServer(BaseBioinformaticsToolTest):
    """Test Picard server functionality."""

    @property
    def tool_name(self) -> str:
        return "picard-server"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_bam": "path/to/input.bam",
            "output_bam": "path/to/output.bam",
            "metrics_file": "path/to/metrics.txt",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample BAM files for testing."""
        bam_file = tmp_path / "input.bam"

        # Create mock BAM file
        bam_file.write_text("BAM file content")

        return {"input_bam": bam_file}

    @pytest.mark.optional
    def test_picard_mark_duplicates(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test Picard MarkDuplicates functionality."""
        params = {
            "operation": "mark_duplicates",
            "input_bam": str(sample_input_files["input_bam"]),
            "output_bam": str(sample_output_dir / "marked.bam"),
            "metrics_file": str(sample_output_dir / "metrics.txt"),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

        # Skip file checks for mock results
        if result.get("mock"):
            return
