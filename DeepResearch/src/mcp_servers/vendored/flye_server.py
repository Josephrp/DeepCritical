"""
Flye MCP Server - Vendored BioinfoMCP server for long-read genome assembly.

This module implements a strongly-typed MCP server for Flye, a de novo assembler
for single-molecule sequencing reads, using Pydantic AI patterns and testcontainers deployment.
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


class FlyeServer(MCPServerBase):
    """MCP Server for Flye long-read genome assembler with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="flye-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"FLYE_VERSION": "2.9.2"},
                capabilities=[
                    "genome_assembly",
                    "long_read_assembly",
                    "nanopore",
                    "pacbio",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def flye_assembly(
        self,
        reads: list[str],
        output_dir: str,
        genome_size: str,
        threads: int = 4,
        iterations: int = 1,
        meta: bool = False,
        plasmid: bool = False,
        trestle: bool = False,
        keep_haplotypes: bool = False,
        scaffold: bool = False,
        resume: bool = False,
        resume_from: str | None = None,
        stop_after: str | None = None,
        read_error: float = 0.0,
        min_overlap: int = 1000,
        asm_coverage: int = 0,
        homo_polymer: bool = False,
    ) -> dict[str, Any]:
        """
        Assemble genome from long-read sequencing data using Flye.

        This tool performs de novo assembly of genomes from long-read sequencing
        technologies like PacBio and Oxford Nanopore.

        Args:
            reads: List of input read files (FASTQ/FASTA format)
            output_dir: Output directory for assembly results
            genome_size: Estimated genome size (e.g., "5m", "2.5g")
            threads: Number of parallel threads
            iterations: Number of polishing iterations
            meta: Perform metagenome assembly
            plasmid: Enable plasmid assembly mode
            trestle: Use Trestle algorithm for highly accurate reads
            keep_haplotypes: Keep haplotype information in output
            scaffold: Enable scaffolding with graph
            resume: Resume from previous interrupted run
            resume_from: Resume from specific stage
            stop_after: Stop assembly after specific stage
            read_error: Expected read error rate (0 = auto-detect)
            min_overlap: Minimum overlap between reads
            asm_coverage: Target assembly coverage
            homo_polymer: Enable homopolymer compression

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        for read_file in reads:
            read_path = Path(read_file)
            if not read_path.exists():
                raise FileNotFoundError(f"Read file not found: {read_file}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        if threads < 1:
            raise ValueError("threads must be >= 1")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if read_error < 0.0 or read_error > 1.0:
            raise ValueError("read_error must be between 0.0 and 1.0")
        if min_overlap < 0:
            raise ValueError("min_overlap must be >= 0")
        if asm_coverage < 0:
            raise ValueError("asm_coverage must be >= 0")

        # Build command
        cmd = [
            "flye",
            "--genome-size",
            genome_size,
            "--out-dir",
            output_dir,
            "--threads",
            str(threads),
        ]

        # Add read files
        for read_file in reads:
            if read_file.endswith((".fastq", ".fq", ".fastq.gz", ".fq.gz")):
                cmd.extend(["--nano-raw", read_file])
            elif read_file.endswith((".fasta", ".fa", ".fasta.gz", ".fa.gz")):
                cmd.extend(["--nano-raw", read_file])  # Flye can handle FASTA too
            else:
                # Assume nanopore reads by default
                cmd.extend(["--nano-raw", read_file])

        # Add optional parameters
        if iterations != 1:
            cmd.extend(["--iterations", str(iterations)])

        if meta:
            cmd.append("--meta")

        if plasmid:
            cmd.append("--plasmid")

        if trestle:
            cmd.append("--trestle")

        if keep_haplotypes:
            cmd.append("--keep-haplotypes")

        if scaffold:
            cmd.append("--scaffold")

        if resume:
            cmd.append("--resume")

        if resume_from:
            cmd.extend(["--resume-from", resume_from])

        if stop_after:
            cmd.extend(["--stop-after", stop_after])

        if read_error > 0.0:
            cmd.extend(["--read-error", str(read_error)])

        if min_overlap != 1000:
            cmd.extend(["--min-overlap", str(min_overlap)])

        if asm_coverage > 0:
            cmd.extend(["--asm-coverage", str(asm_coverage)])

        if homo_polymer:
            cmd.append("--homo-polymer")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=86400,  # 24 hours timeout for assembly
                cwd=output_dir,
            )

            # Check for expected output files
            output_files = []
            assembly_file = output_path / "assembly.fasta"
            if assembly_file.exists():
                output_files.append(str(assembly_file))

            graph_file = output_path / "assembly_graph.gfa"
            if graph_file.exists():
                output_files.append(str(graph_file))

            info_file = output_path / "assembly_info.txt"
            if info_file.exists():
                output_files.append(str(info_file))

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
                "error": f"Flye assembly failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Flye assembly timed out after 24 hours",
            }
