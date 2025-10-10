"""
TopHat server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestTopHatServer(BaseBioinformaticsToolTest):
    """Test TopHat server functionality."""

    @property
    def tool_name(self) -> str:
        return "tophat"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "index": "path/to/index",
            "mate1": "path/to/reads_1.fq",
            "output_dir": "path/to/output",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTQ files for testing."""
        reads_file = tmp_path / "reads_1.fq"

        # Create mock FASTQ file
        reads_file.write_text("@read1\nATCGATCGATCG\n+\nIIIIIIIIIIII\n")

        return {"mate1": reads_file}

    @pytest.mark.optional
    def test_tophat_align(self, tool_instance, sample_input_files, sample_output_dir):
        """Test TopHat align functionality."""
        params = {
            "index": "test_index",
            "mate1": str(sample_input_files["mate1"]),
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
