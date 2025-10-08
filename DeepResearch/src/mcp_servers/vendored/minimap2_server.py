"""
Minimap2 MCP Server - Vendored BioinfoMCP server for versatile pairwise alignment.

This module implements a strongly-typed MCP server for Minimap2, a versatile
pairwise aligner for nucleotide and long-read sequencing technologies,
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


class Minimap2Server(MCPServerBase):
    """MCP Server for Minimap2 versatile pairwise aligner with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="minimap2-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"MINIMAP2_VERSION": "2.26"},
                capabilities=[
                    "sequence_alignment",
                    "long_read_alignment",
                    "genome_alignment",
                    "nanopore",
                    "pacbio",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def minimap2_align(
        self,
        target: str,
        query: list[str],
        output_sam: str,
        preset: str = "map-ont",
        threads: int = 4,
        output_format: str = "sam",
        secondary_alignments: bool = True,
        max_fragment_length: int = 800,
        min_chain_score: int = 40,
        min_dp_score: int = 40,
        min_matching_length: int = 40,
        bandwidth: int = 500,
        zdrop_score: int = 400,
        min_occ_floor: int = 100,
        chain_gap_scale: float = 0.3,
        match_score: int = 2,
        mismatch_penalty: int = 4,
        gap_open_penalty: int = 4,
        gap_extension_penalty: int = 2,
        prune_factor: int = 10,
    ) -> dict[str, Any]:
        """
        Align sequences using Minimap2 versatile pairwise aligner.

        This tool performs sequence alignment optimized for various sequencing
        technologies including Oxford Nanopore, PacBio, and Illumina reads.

        Args:
            target: Target sequence file (FASTA/FASTQ)
            query: Query sequence files (FASTA/FASTQ)
            output_sam: Output alignment file (SAM/BAM format)
            preset: Alignment preset (map-ont, map-pb, map-hifi, sr, splice, etc.)
            threads: Number of threads
            output_format: Output format (sam, bam, paf)
            secondary_alignments: Report secondary alignments
            max_fragment_length: Maximum fragment length for SR mode
            min_chain_score: Minimum chaining score
            min_dp_score: Minimum DP alignment score
            min_matching_length: Minimum matching length
            bandwidth: Chaining bandwidth
            zdrop_score: Z-drop score for alignment termination
            min_occ_floor: Minimum occurrence floor
            chain_gap_scale: Chain gap scale factor
            match_score: Match score
            mismatch_penalty: Mismatch penalty
            gap_open_penalty: Gap open penalty
            gap_extension_penalty: Gap extension penalty
            prune_factor: Prune factor for DP

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        target_path = Path(target)
        if not target_path.exists():
            raise FileNotFoundError(f"Target file not found: {target}")

        for query_file in query:
            query_path = Path(query_file)
            if not query_path.exists():
                raise FileNotFoundError(f"Query file not found: {query_file}")

        # Validate parameters
        if threads < 1:
            raise ValueError("threads must be >= 1")
        if max_fragment_length <= 0:
            raise ValueError("max_fragment_length must be > 0")
        if min_chain_score < 0:
            raise ValueError("min_chain_score must be >= 0")
        if min_dp_score < 0:
            raise ValueError("min_dp_score must be >= 0")
        if min_matching_length < 0:
            raise ValueError("min_matching_length must be >= 0")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")
        if zdrop_score < 0:
            raise ValueError("zdrop_score must be >= 0")
        if min_occ_floor < 0:
            raise ValueError("min_occ_floor must be >= 0")
        if chain_gap_scale <= 0:
            raise ValueError("chain_gap_scale must be > 0")
        if match_score < 0:
            raise ValueError("match_score must be >= 0")
        if mismatch_penalty < 0:
            raise ValueError("mismatch_penalty must be >= 0")
        if gap_open_penalty < 0:
            raise ValueError("gap_open_penalty must be >= 0")
        if gap_extension_penalty < 0:
            raise ValueError("gap_extension_penalty must be >= 0")
        if prune_factor < 1:
            raise ValueError("prune_factor must be >= 1")

        # Build command
        cmd = [
            "minimap2",
            "-x",
            preset,
            "-t",
            str(threads),
            "-a",  # Output SAM format
        ]

        # Add output format option
        if output_format == "bam":
            cmd.extend(["-o", output_sam + ".tmp.sam"])
        else:
            cmd.extend(["-o", output_sam])

        # Add secondary alignments option
        if not secondary_alignments:
            cmd.extend(["-N", "1"])

        # Add scoring parameters
        cmd.extend(
            [
                "-A",
                str(match_score),
                "-B",
                str(mismatch_penalty),
                "-O",
                f"{gap_open_penalty},{gap_extension_penalty}",
                "-E",
                f"{gap_open_penalty},{gap_extension_penalty}",
                "-z",
                str(zdrop_score),
                "-s",
                str(min_chain_score),
                "-u",
                str(min_dp_score),
                "-L",
                str(min_matching_length),
                "-f",
                str(min_occ_floor),
                "-r",
                str(max_fragment_length),
                "-g",
                str(bandwidth),
                "-p",
                str(chain_gap_scale),
                "-M",
                str(prune_factor),
            ]
        )

        # Add target and query files
        cmd.append(target)
        cmd.extend(query)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=3600
            )

            # Convert SAM to BAM if requested
            output_files = []
            if output_format == "bam":
                # Convert SAM to BAM
                bam_cmd = [
                    "samtools",
                    "view",
                    "-b",
                    "-o",
                    output_sam,
                    output_sam + ".tmp.sam",
                ]
                try:
                    subprocess.run(bam_cmd, check=True, capture_output=True)
                    Path(output_sam + ".tmp.sam").unlink(missing_ok=True)
                    if Path(output_sam).exists():
                        output_files.append(output_sam)
                except subprocess.CalledProcessError:
                    # If conversion fails, keep the SAM file
                    Path(output_sam + ".tmp.sam").rename(output_sam)
                    output_files.append(output_sam)
            elif Path(output_sam).exists():
                output_files.append(output_sam)

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
                "error": f"Minimap2 alignment failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Minimap2 alignment timed out after 3600 seconds",
            }
