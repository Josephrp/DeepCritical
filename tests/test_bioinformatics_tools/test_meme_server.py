"""
MEME server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestMEMEServer(BaseBioinformaticsToolTest):
    """Test MEME server functionality."""

    @property
    def tool_name(self) -> str:
        return "meme"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "sequences": "path/to/sequences.fa",
            "output_dir": "path/to/output",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTA files for testing."""
        sequences_file = tmp_path / "sequences.fa"

        # Create mock FASTA file
        sequences_file.write_text(
            ">seq1\n"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
            ">seq2\n"
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n"
        )

        return {"sequences": sequences_file}

    @pytest.mark.optional
    def test_meme_motif_discovery(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test MEME motif discovery functionality."""
        params = {
            "sequences": str(sample_input_files["sequences"]),
            "output_dir": str(sample_output_dir),
            "nmotifs": 1,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
