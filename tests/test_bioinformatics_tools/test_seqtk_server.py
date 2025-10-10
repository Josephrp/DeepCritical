"""
Seqtk server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestSeqtkServer(BaseBioinformaticsToolTest):
    """Test Seqtk server functionality."""

    @property
    def tool_name(self) -> str:
        return "seqtk"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_file": "path/to/sequences.fa",
            "fraction": 0.1,
            "output_file": "path/to/sampled.fa",
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

        return {"input_file": sequences_file}

    @pytest.mark.optional
    def test_seqtk_sample(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Seqtk sample functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "fraction": 0.1,
            "output_file": str(sample_output_dir / "sampled.fa"),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
