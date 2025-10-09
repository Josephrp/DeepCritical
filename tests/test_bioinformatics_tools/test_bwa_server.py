"""
BWA server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from DeepResearch.src.tools.bioinformatics.bwa_server import BWAServer
from tests.utils.mocks.mock_data import create_mock_fasta, create_mock_fastq


class TestBWAServer:
    """Test BWA server functionality."""

    @pytest.fixture
    def bwa_server(self):
        """Create BWA server instance for testing."""
        return BWAServer()

    @pytest.fixture
    def sample_fastq(self, tmp_path):
        """Create sample FASTQ file for testing."""
        return create_mock_fastq(tmp_path / "sample.fq", num_reads=100)

    @pytest.fixture
    def sample_fasta(self, tmp_path):
        """Create sample FASTA file for testing."""
        return create_mock_fasta(tmp_path / "reference.fa", num_sequences=10)

    def test_bwa_index_creation(self, bwa_server, tmp_path, sample_fasta):
        """Test BWA index creation functionality."""
        index_dir = tmp_path / "bwa_index"

        result = bwa_server.bwa_index(
            reference=str(sample_fasta),
            prefix=str(index_dir / "test_index"),
            algorithm="bwtsw",
        )

        assert result["success"]
        assert "command_executed" in result
        assert len(result["output_files"]) > 0

        # Verify index files were created
        for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]:
            index_file = index_dir / f"test_index{ext}"
            assert index_file.exists()

    def test_bwa_alignment(self, bwa_server, tmp_path, sample_fastq, sample_fasta):
        """Test BWA alignment functionality."""
        # Create index first
        index_dir = tmp_path / "index"
        index_result = bwa_server.bwa_index(
            reference=str(sample_fasta),
            prefix=str(index_dir / "ref_index"),
            algorithm="bwtsw",
        )
        assert index_result["success"]

        # Test alignment
        output_sam = tmp_path / "alignment.sam"
        align_result = bwa_server.bwa_align(
            index=str(index_dir / "ref_index"),
            read1=str(sample_fastq),
            output_file=str(output_sam),
            threads=1,
        )

        assert align_result["success"]
        assert output_sam.exists()

        # Verify SAM file has content
        content = output_sam.read_text()
        assert "@SQ" in content  # Header line
        assert len(content.split("\n")) > 10  # Multiple lines

    def test_error_handling(self, bwa_server):
        """Test error handling for invalid inputs."""
        result = bwa_server.bwa_index(
            reference="/nonexistent/file.fa",
            prefix="/tmp/test_index",
            algorithm="bwtsw",
        )

        assert not result["success"]
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.containerized
    async def test_containerized_bwa_workflow(
        self, bwa_server, tmp_path, sample_fastq, sample_fasta
    ):
        """Test complete BWA workflow in containerized environment."""
        # Deploy server in container
        deployment = await bwa_server.deploy_with_testcontainers()
        assert deployment.status == "running"

        try:
            # Execute full workflow in container
            index_result = bwa_server.bwa_index(
                reference=str(sample_fasta),
                index_basename=str(tmp_path / "container_index"),
            )

            align_result = bwa_server.bwa_align(
                index=str(tmp_path / "container_index"),
                read1=str(sample_fastq),
                output_file=str(tmp_path / "container_alignment.sam"),
            )

            assert index_result["success"]
            assert align_result["success"]

        finally:
            # Cleanup
            await bwa_server.stop_with_testcontainers()
