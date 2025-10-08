"""
Qualimap MCP Server - Vendored BioinfoMCP server for quality control and assessment.

This module implements a strongly-typed MCP server for Qualimap, a tool for quality
control and assessment of sequencing data, using Pydantic AI patterns and testcontainers deployment.
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


class QualimapServer(MCPServerBase):
    """MCP Server for Qualimap quality control and assessment tools with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="qualimap-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"QUALIMAP_VERSION": "2.3"},
                capabilities=[
                    "quality_control",
                    "bam_qc",
                    "rna_seq_qc",
                    "alignment_assessment",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def qualimap_bamqc(
        self,
        bam_file: str,
        output_dir: str,
        java_mem_size: str = "4G",
        paint_chromosome_limits: bool = False,
        genome_gc_distr: str | None = None,
        feature_file: str | None = None,
        skip_duplicated: bool = False,
        skip_dup_mode: int = 0,
        algorithm: str = "unique-3",
        min_mapq: int = 0,
        skip_secondary: bool = False,
        skip_supplementary: bool = False,
        use_equal_mapq: bool = False,
        count_duplicates: bool = False,
        validate_sam: bool = False,
    ) -> dict[str, Any]:
        """
        Perform quality control analysis on BAM files using Qualimap.

        This tool analyzes BAM files to provide comprehensive quality control
        metrics including coverage, mapping quality, GC content, and more.

        Args:
            bam_file: Input BAM file
            output_dir: Output directory for results
            java_mem_size: Java heap size (e.g., "4G", "8G")
            paint_chromosome_limits: Paint chromosome limits in coverage plots
            genome_gc_distr: Genome GC distribution file
            feature_file: Feature file (GTF/GFF) for gene coverage analysis
            skip_duplicated: Skip duplicated alignments
            skip_dup_mode: Skip duplicates mode (0=count all, 1=skip optical, 2=skip all)
            algorithm: Counting algorithm for feature assignment
            min_mapq: Minimum mapping quality
            skip_secondary: Skip secondary alignments
            skip_supplementary: Skip supplementary alignments
            use_equal_mapq: Use equal mapping quality for all alignments
            count_duplicates: Count duplicate reads
            validate_sam: Validate SAM/BAM file

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input file
        bam_path = Path(bam_file)
        if not bam_path.exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        if min_mapq < 0:
            raise ValueError("min_mapq must be >= 0")
        if skip_dup_mode not in [0, 1, 2]:
            raise ValueError("skip_dup_mode must be 0, 1, or 2")

        # Build command
        cmd = [
            "qualimap",
            "bamqc",
            "-bam",
            bam_file,
            "-outdir",
            output_dir,
            "-java-mem-size=" + java_mem_size,
        ]

        # Add optional parameters
        if paint_chromosome_limits:
            cmd.append("-paint-chromosome-limits")

        if genome_gc_distr:
            gc_path = Path(genome_gc_distr)
            if not gc_path.exists():
                raise FileNotFoundError(
                    f"Genome GC distribution file not found: {genome_gc_distr}"
                )
            cmd.extend(["-genome-gc-distr", genome_gc_distr])

        if feature_file:
            feature_path = Path(feature_file)
            if not feature_path.exists():
                raise FileNotFoundError(f"Feature file not found: {feature_file}")
            cmd.extend(["-gff", feature_file])

        if skip_duplicated:
            cmd.append("-skip-duplicated")

        if skip_dup_mode != 0:
            cmd.extend(["-skip-dup-mode", str(skip_dup_mode)])

        if algorithm != "unique-3":
            cmd.extend(["-algorithm", algorithm])

        if min_mapq > 0:
            cmd.extend(["--min-mapq", str(min_mapq)])

        if skip_secondary:
            cmd.append("--skip-secondary")

        if skip_supplementary:
            cmd.append("--skip-supplementary")

        if use_equal_mapq:
            cmd.append("--use-equal-mapq")

        if count_duplicates:
            cmd.append("-count-duplicates")

        if validate_sam:
            cmd.append("--validate-sam")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 minutes timeout
            )

            # Check for expected output files
            output_files = []
            report_file = output_path / "qualimapReport.html"
            if report_file.exists():
                output_files.append(str(report_file))

            stats_file = output_path / "genome_results.txt"
            if stats_file.exists():
                output_files.append(str(stats_file))

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
                "error": f"Qualimap BAM QC failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Qualimap BAM QC timed out after 1800 seconds",
            }

    @mcp_tool()
    def qualimap_rnaseq(
        self,
        bam_file: str,
        gtf_file: str,
        output_dir: str,
        java_mem_size: str = "4G",
        algorithm: str = "uniquely-mapped-reads",
        paired: bool = False,
        strand_specific: str = "non-strand-specific",
        sorted: bool = False,
        out_format: str = "HTML",
        skip_duplicated: bool = False,
        min_mapq: int = 0,
    ) -> dict[str, Any]:
        """
        Perform RNA-seq quality control analysis using Qualimap.

        This tool analyzes RNA-seq BAM files to provide quality control metrics
        including gene coverage, junction analysis, and transcript-level statistics.

        Args:
            bam_file: Input BAM file from RNA-seq alignment
            gtf_file: Gene annotation file (GTF format)
            output_dir: Output directory for results
            java_mem_size: Java heap size (e.g., "4G", "8G")
            algorithm: Counting algorithm for gene assignment
            paired: Input data is paired-end
            strand_specific: Strand specificity (non-strand-specific, strand-specific-forward, strand-specific-reverse)
            sorted: Input BAM is coordinate-sorted
            out_format: Output format (HTML, PDF)
            skip_duplicated: Skip duplicated alignments
            min_mapq: Minimum mapping quality

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        bam_path = Path(bam_file)
        gtf_path = Path(gtf_file)
        if not bam_path.exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")
        if not gtf_path.exists():
            raise FileNotFoundError(f"GTF file not found: {gtf_file}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        if min_mapq < 0:
            raise ValueError("min_mapq must be >= 0")
        if strand_specific not in [
            "non-strand-specific",
            "strand-specific-forward",
            "strand-specific-reverse",
        ]:
            raise ValueError(
                "strand_specific must be one of: non-strand-specific, strand-specific-forward, strand-specific-reverse"
            )
        if out_format not in ["HTML", "PDF"]:
            raise ValueError("out_format must be HTML or PDF")

        # Build command
        cmd = [
            "qualimap",
            "rnaseq",
            "-bam",
            bam_file,
            "-gtf",
            gtf_file,
            "-outdir",
            output_dir,
            "-java-mem-size=" + java_mem_size,
            "-algorithm",
            algorithm,
            "-outformat",
            out_format,
        ]

        if paired:
            cmd.append("-pe")

        if strand_specific != "non-strand-specific":
            if (
                strand_specific == "strand-specific-forward"
                or strand_specific == "strand-specific-reverse"
            ):
                cmd.append("-s")

        if sorted:
            cmd.append("--sorted")

        if skip_duplicated:
            cmd.append("-skip-duplicated")

        if min_mapq > 0:
            cmd.extend(["--min-mapq", str(min_mapq)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )

            # Check for expected output files
            output_files = []
            report_file = output_path / "qualimapReport.html"
            if report_file.exists():
                output_files.append(str(report_file))

            stats_file = output_path / "rnaseq_qc_results.txt"
            if stats_file.exists():
                output_files.append(str(stats_file))

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
                "error": f"Qualimap RNA-seq QC failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Qualimap RNA-seq QC timed out after 3600 seconds",
            }
