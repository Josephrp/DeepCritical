"""
MultiQC server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestMultiQCServer(BaseBioinformaticsToolTest):
    """Test MultiQC server functionality."""

    @property
    def tool_name(self) -> str:
        return "multiqc"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_dir": "path/to/analysis_results",
            "output_dir": "path/to/output",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample analysis results for testing."""
        input_dir = tmp_path / "analysis_results"
        input_dir.mkdir()

        # Create mock analysis files
        fastqc_file = input_dir / "sample_fastqc.zip"
        fastqc_file.write_text("FastQC analysis results")

        return {"input_dir": input_dir}

    @pytest.mark.optional
    def test_multiqc_run(self, tool_instance, sample_input_files, sample_output_dir):
        """Test MultiQC run functionality."""
        params = {
            "input_dir": str(sample_input_files["input_dir"]),
            "output_dir": str(sample_output_dir),
            "filename": "multiqc_report",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
