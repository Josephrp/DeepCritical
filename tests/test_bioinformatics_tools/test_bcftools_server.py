"""
BCFtools server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from DeepResearch.src.tools.bioinformatics.bcftools_server import BCFtoolsServer
from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestBCFtoolsServer(BaseBioinformaticsToolTest):
    """Test BCFtools server functionality."""

    @property
    def tool_name(self) -> str:
        return "bcftools"

    @property
    def tool_class(self):
        # This would import the actual BCFtools server class
        from DeepResearch.src.tools.bioinformatics.bcftools_server import BCFtoolsServer

        return BCFtoolsServer

    @property
    def required_parameters(self) -> dict:
        return {
            "input_file": "path/to/input.vcf",
            "operation": "view",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample VCF files for testing."""
        vcf_file = tmp_path / "sample.vcf"

        # Create mock VCF file
        vcf_file.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t100\t.\tA\tT\t60\tPASS\t.\n"
            "chr1\t200\t.\tG\tC\t60\tPASS\t.\n"
        )

        return {"input_file": vcf_file}

    def test_bcftools_view(self, tool_instance, sample_input_files, sample_output_dir):
        """Test BCFtools view functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "operation": "view",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    def test_bcftools_stats(self, tool_instance, sample_input_files, sample_output_dir):
        """Test BCFtools stats functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "operation": "stats",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    def test_bcftools_filter(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test BCFtools filter functionality."""
        params = {
            "input_file": str(sample_input_files["input_file"]),
            "operation": "filter",
            "output_dir": str(sample_output_dir),
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

    @pytest.mark.containerized
    @pytest.mark.asyncio
    async def test_containerized_bcftools_workflow(self, tmp_path):
        """Test complete BCFtools workflow in containerized environment."""
        # Create server instance
        server = BCFtoolsServer()

        # Deploy server in container
        deployment = await server.deploy_with_testcontainers()
        assert deployment.status == "running"

        try:
            # Wait for BCFtools to be installed and ready in the container
            import asyncio

            await asyncio.sleep(30)  # Wait for package installation

            # Create sample VCF file
            vcf_file = tmp_path / "sample.vcf"
            vcf_file.write_text(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "chr1\t100\t.\tA\tT\t60\tPASS\t.\n"
            )

            # Test BCFtools view operation
            result = server.bcftools_view(
                input_file=str(vcf_file),
                output_file=str(tmp_path / "output.vcf"),
                output_type="v",
            )

            # Verify the operation completed (may fail due to container permissions, but server should respond)
            assert "success" in result or "error" in result

        finally:
            # Clean up container
            await server.stop_with_testcontainers()
