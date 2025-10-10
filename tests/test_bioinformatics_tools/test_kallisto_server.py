"""
Kallisto server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestKallistoServer(BaseBioinformaticsToolTest):
    """Test Kallisto server functionality."""

    @property
    def tool_name(self) -> str:
        return "kallisto"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "transcripts": "path/to/transcripts.fa",
            "index": "path/to/index",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample FASTA files for testing."""
        transcripts_file = tmp_path / "transcripts.fa"

        # Create mock FASTA file
        transcripts_file.write_text(
            ">transcript1\n"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
            ">transcript2\n"
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n"
        )

        return {"transcripts": transcripts_file}

    @pytest.mark.optional
    def test_kallisto_index(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Kallisto index functionality."""
        params = {
            "transcripts": str(sample_input_files["transcripts"]),
            "index": str(sample_output_dir / "kallisto_index"),
            "kmer_len": 31,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.optional
    def test_kallisto_quant(self, tool_instance, sample_input_files, sample_output_dir):
        """Test Kallisto quant functionality."""
        params = {
            "index": str(sample_output_dir / "kallisto_index"),
            "lib_type": "A",
            "mates1": [str(sample_input_files["transcripts"])],
            "output": str(sample_output_dir / "quant"),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result
