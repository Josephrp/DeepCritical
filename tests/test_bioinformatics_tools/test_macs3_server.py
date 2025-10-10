"""
MACS3 server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestMACS3Server(BaseBioinformaticsToolTest):
    """Test MACS3 server functionality."""

    @property
    def tool_name(self) -> str:
        return "macs3"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "treatment_file": "path/to/treatment.bam",
            "output_dir": "path/to/output",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample BAM files for testing."""
        bam_file = tmp_path / "treatment.bam"

        # Create mock BAM file
        bam_file.write_text("BAM file content")

        return {"treatment_file": bam_file}

    @pytest.mark.optional
    def test_macs3_callpeak(self, tool_instance, sample_input_files, sample_output_dir):
        """Test MACS3 callpeak functionality."""
        params = {
            "treatment_file": str(sample_input_files["treatment_file"]),
            "output_dir": str(sample_output_dir),
            "name": "peaks",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
