"""
BUSCO server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestBUSCOServer(BaseBioinformaticsToolTest):
    """Test BUSCO server functionality."""

    @property
    def tool_name(self) -> str:
        return "busco"

    @property
    def tool_class(self):
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_file": "path/to/genome.fa",
            "output_dir": "path/to/output",
            "mode": "genome",
            "lineage_dataset": "bacteria_odb10",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample genome files for testing."""
        genome_file = tmp_path / "sample_genome.fa"

        # Create mock FASTA file
        genome_file.write_text(
            ">contig1\n"
            "ATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
            ">contig2\n"
            "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n"
        )

        return {"input_file": genome_file}

    @pytest.mark.optional
    def test_busco_run(self, tool_instance, sample_input_files, sample_output_dir):
        """Test BUSCO run functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "output_dir": str(sample_output_dir),
            "mode": "genome",
            "lineage_dataset": "bacteria_odb10",
            "cpu": 1,
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.optional
    def test_busco_download(self, tool_instance, sample_output_dir):
        """Test BUSCO download functionality."""
        params = {
            "lineage_dataset": "bacteria_odb10",
            "download_path": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
