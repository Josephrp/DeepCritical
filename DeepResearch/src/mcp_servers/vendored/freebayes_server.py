"""
FreeBayes MCP Server - Vendored BioinfoMCP server for Bayesian haplotype-based variant calling.

This module implements a strongly-typed MCP server for FreeBayes, a Bayesian genetic
variant detector designed to find small polymorphisms, specifically SNPs, indels,
MNPs, and complex events smaller than the length of a short-read sequencing alignment,
using Pydantic AI patterns and testcontainers deployment.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ...datatypes.mcp import (
    MCPAgentIntegration,
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)
from ..utils.mcp_server_base import MCPServerBase, mcp_tool


class FreeBayesServer(MCPServerBase):
    """MCP Server for FreeBayes Bayesian haplotype-based variant calling with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="freebayes-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"FREEBAYES_VERSION": "1.3.6"},
                capabilities=[
                    "variant_calling",
                    "snp_calling",
                    "indel_calling",
                    "genomics",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def freebayes_variant_calling(
        self,
        bam_files: list[str],
        reference_fasta: str,
        output_vcf: str,
        region: str | None = None,
        targets_bed: str | None = None,
        min_mapping_quality: int = 1,
        min_base_quality: int = 3,
        min_alternate_fraction: float = 0.2,
        min_alternate_count: int = 2,
        ploidy: int = 2,
        theta: float = 0.001,
        pooled_discrete: bool = False,
        pooled_continuous: bool = False,
        report_monomorphic: bool = False,
        genotype_qualities: bool = True,
        strict_vcf: bool = False,
    ) -> dict[str, Any]:
        """
        Call genetic variants using FreeBayes Bayesian haplotype-based approach.

        This tool performs variant calling on BAM files to identify SNPs, indels,
        and other small variants using a Bayesian statistical framework.

        Args:
            bam_files: List of input BAM files to analyze
            reference_fasta: Reference genome FASTA file (must be indexed)
            output_vcf: Output VCF file path
            region: Genomic region to analyze (chr:start-end format)
            targets_bed: BED file specifying target regions
            min_mapping_quality: Minimum mapping quality score
            min_base_quality: Minimum base quality score
            min_alternate_fraction: Minimum fraction of alternate observations
            min_alternate_count: Minimum count of alternate observations
            ploidy: Sample ploidy (default: diploid)
            theta: Expected mutation rate
            pooled_discrete: Use discrete pooled model
            pooled_continuous: Use continuous pooled model
            report_monomorphic: Report monomorphic loci
            genotype_qualities: Calculate genotype qualities
            strict_vcf: Generate strict VCF format

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        reference_path = Path(reference_fasta)
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference FASTA file not found: {reference_fasta}"
            )

        for bam_file in bam_files:
            bam_path = Path(bam_file)
            if not bam_path.exists():
                raise FileNotFoundError(f"BAM file not found: {bam_file}")

        if targets_bed:
            targets_path = Path(targets_bed)
            if not targets_path.exists():
                raise FileNotFoundError(f"Targets BED file not found: {targets_bed}")

        # Validate parameters
        if min_mapping_quality < 0:
            raise ValueError("min_mapping_quality must be >= 0")
        if min_base_quality < 0:
            raise ValueError("min_base_quality must be >= 0")
        if not (0.0 <= min_alternate_fraction <= 1.0):
            raise ValueError("min_alternate_fraction must be between 0.0 and 1.0")
        if min_alternate_count < 0:
            raise ValueError("min_alternate_count must be >= 0")
        if ploidy < 1:
            raise ValueError("ploidy must be >= 1")
        if theta < 0.0:
            raise ValueError("theta must be >= 0.0")

        # Build command
        cmd = ["freebayes", "-f", reference_fasta]

        # Add BAM files
        for bam_file in bam_files:
            cmd.extend(["-b", bam_file])

        # Add output
        cmd.extend(["-v", output_vcf])

        # Add optional parameters
        if region:
            cmd.extend(["-r", region])

        if targets_bed:
            cmd.extend(["-t", targets_bed])

        if min_mapping_quality != 1:
            cmd.extend(["-m", str(min_mapping_quality)])

        if min_base_quality != 3:
            cmd.extend(["-q", str(min_base_quality)])

        if min_alternate_fraction != 0.2:
            cmd.extend(["-F", str(min_alternate_fraction)])

        if min_alternate_count != 2:
            cmd.extend(["-C", str(min_alternate_count)])

        if ploidy != 2:
            cmd.extend(["-p", str(ploidy)])

        if theta != 0.001:
            cmd.extend(["-T", str(theta)])

        if pooled_discrete:
            cmd.append("-J")

        if pooled_continuous:
            cmd.append("-K")

        if report_monomorphic:
            cmd.append("--report-monomorphic")

        if genotype_qualities:
            cmd.append("-=")

        if strict_vcf:
            cmd.append("--strict-vcf")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=3600
            )

            output_files = []
            if Path(output_vcf).exists():
                output_files.append(output_vcf)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "success": False,
                "error": f"FreeBayes failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "FreeBayes timed out after 3600 seconds",
            }
