"""
Minimap2 server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestMinimap2Server(BaseBioinformaticsToolTest):
    """Test Minimap2 server functionality."""

    @property
    def tool_name(self) -> str:
        return "minimap2"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "target": "path/to/reference.fa",
            "query": ["path/to/reads.fq"],
            "output_sam": "path/to/output.sam",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTA/FASTQ files for testing."""
        reference_file = tmp_path / "reference.fa"
        reads_file = tmp_path / "reads.fq"

        # Create mock FASTA file
        reference_file.write_text(">chr1\nATCGATCGATCGATCGATCGATCGATCGATCGATCG\n")

        # Create mock FASTQ file
        reads_file.write_text("@read1\nATCGATCGATCG\n+\nIIIIIIIIIIII\n")

        return {"target": reference_file, "query": [reads_file]}

    @pytest.mark.optional
    def test_minimap2_align(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Minimap2 align functionality."""
        params = {
            "target": str(sample_input_files["target"]),
            "query": [str(sample_input_files["query"][0])],
            "output_sam": str(sample_output_dir / "aligned.sam"),
            "preset": "map-ont",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
