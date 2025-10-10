"""
HOMER server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestHOMERServer(BaseBioinformaticsToolTest):
    """Test HOMER server functionality."""

    @property
    def tool_name(self) -> str:
        return "homer"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_file": "path/to/peaks.bed",
            "output_dir": "path/to/output",
            "genome": "hg38",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample BED files for testing."""
        peaks_file = tmp_path / "peaks.bed"

        # Create mock BED file
        peaks_file.write_text("chr1\t100\t200\tpeak1\t10\nchr1\t300\t400\tpeak2\t8\n")

        return {"input_file": peaks_file}

    @pytest.mark.optional
    def test_homer_findMotifs(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test HOMER findMotifs functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "output_dir": str(sample_output_dir),
            "genome": "hg38",
            "size": "200",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.optional
    def test_homer_annotatePeaks(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test HOMER annotatePeaks functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "genome": "hg38",
            "output_file": str(sample_output_dir / "annotated.txt"),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
