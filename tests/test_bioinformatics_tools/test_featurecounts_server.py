"""
featureCounts server component tests.
"""

import tempfile
from pathlib import Path

import pytest

from tests.test_bioinformatics_tools.base.test_base_tool import (
    BaseBioinformaticsToolTest,
)


class TestFeatureCountsServer(BaseBioinformaticsToolTest):
    """Test featureCounts server functionality."""

    @property
    def tool_name(self) -> str:
        return "featurecounts"

    @property
    def tool_class(self):
        # This would import the actual featureCounts server class
        from unittest.mock import Mock

        return Mock

    @property
    def required_parameters(self) -> dict:
        return {
            "input_files": "path/to/aligned.bam",
            "annotation": "path/to/genes.gtf",
            "output_file": "counts.txt",
        }

    @pytest.fixture
    def sample_input_files(self, tmp_path):
        """Create sample BAM and GTF files for testing."""
        bam_file = tmp_path / "aligned.bam"
        gtf_file = tmp_path / "genes.gtf"

        # Create mock BAM file
        bam_file.write_bytes(b"mock bam content")

        # Create mock GTF annotation
        gtf_file.write_text(
            'chr1\tHAVANA\tgene\t1\t1000\t.\t+\t.\tgene_id "GENE1"; gene_name "Gene1";\n'
        )

        return {"bam_file": bam_file, "annotation": gtf_file}

    @pytest.mark.optional
    def test_featurecounts_quantification(
        self, tool_instance, sample_input_files, sample_output_dir
    ):
        """Test featureCounts quantification functionality."""
        params = {
            "input_files": str(sample_input_files["bam_file"]),
            "annotation": str(sample_input_files["annotation"]),
            "output_file": str(sample_output_dir / "counts.txt"),
            "feature_type": "gene",
            "attribute_type": "gene_id",
        }

        result = tool_instance.run(params)

        assert result["success"] is True
        assert "output_files" in result

        # Verify counts output file was created
        counts_file = sample_output_dir / "counts.txt"
        assert counts_file.exists()

        # Verify counts format (tab-separated)
        content = counts_file.read_text()
        assert "Geneid" in content  # featureCounts header
