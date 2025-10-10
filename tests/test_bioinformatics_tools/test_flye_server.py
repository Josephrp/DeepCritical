"""
Flye server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestFlyeServer(BaseBioinformaticsToolTest):
    """Test Flye server functionality."""

    @property
    def tool_name(self) -> str:
        return "flye"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "reads": ["path/to/reads.fq"],
            "output_dir": "path/to/output",
            "genome_size": "5m",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTQ files for testing."""
        reads_file = tmp_path / "sample_reads.fq"

        # Create mock FASTQ file
        reads_file.write_text(
            "@read1\nATCGATCGATCGATCGATCGATCGATCGATCGATCG\n+\nIIIIIIIIIIIIIII\n"
        )

        return {"reads": [reads_file]}

    @pytest.mark.optional
    def test_flye_assembly(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Flye assembly functionality."""
        params = {
            "reads": [str(sample_input_files["reads"][0])],
            "output_dir": str(sample_output_dir),
            "genome_size": "5m",
            "threads": 1,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
