"""
TopHat MCP Server - Vendored BioinfoMCP server for RNA-seq alignment.

This module implements a strongly-typed MCP server for TopHat, a fast splice junction
mapper for RNA-Seq reads, using Pydantic AI patterns and testcontainers deployment.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...datatypes.mcp import (
    MCPAgentIntegration,
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)
from ..utils.mcp_server_base import MCPServerBase, mcp_tool


class TopHatServer(MCPServerBase):
    """MCP Server for TopHat RNA-seq splice-aware aligner with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="tophat-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"TOPHAT_VERSION": "2.1.1"},
                capabilities=["rna_seq", "alignment", "splice_aware"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="tophat_align",
            description="Align RNA-seq reads to reference genome using TopHat",
            inputs={
                "index": "str",
                "mate1": "str",
                "mate2": "str | None",
                "output_dir": "str",
                "threads": "int",
                "library_type": "str",
                "mate_inner_dist": "int",
                "mate_std_dev": "int",
                "min_anchor_length": "int",
                "splice_mismatches": "int",
                "min_intron_length": "int",
                "max_intron_length": "int",
                "max_multihits": "int",
                "segment_mismatches": "int",
                "segment_length": "int",
                "no_novel_juncs": "bool",
                "no_gtf_juncs": "bool",
                "transcriptome_only": "bool",
                "microexon_search": "bool",
            },
            outputs={
                "command_executed": "str",
                "stdout": "str",
                "stderr": "str",
                "output_files": "list[str]",
                "exit_code": "int",
            },
            server_type=MCPServerType.CUSTOM,
            examples=[
                {
                    "description": "Align paired-end RNA-seq reads with TopHat",
                    "parameters": {
                        "index": "/data/hg38_index",
                        "mate1": "/data/read1.fq",
                        "mate2": "/data/read2.fq",
                        "output_dir": "/results",
                        "threads": 4,
                        "library_type": "fr-unstranded",
                    },
                }
            ],
        )
    )
    def tophat_align(
        self,
        index: str,
        mate1: str,
        mate2: str | None = None,
        output_dir: str = ".",
        threads: int = 1,
        library_type: str = "fr-unstranded",
        mate_inner_dist: int = 50,
        mate_std_dev: int = 20,
        min_anchor_length: int = 8,
        splice_mismatches: int = 0,
        min_intron_length: int = 70,
        max_intron_length: int = 500000,
        max_multihits: int = 20,
        segment_mismatches: int = 2,
        segment_length: int = 25,
        no_novel_juncs: bool = False,
        no_gtf_juncs: bool = False,
        transcriptome_only: bool = False,
        microexon_search: bool = False,
    ) -> dict[str, Any]:
        """
        Align RNA-seq reads to reference genome using TopHat.

        TopHat is a fast splice junction mapper for RNA-Seq reads. It aligns RNA-Seq reads
        to mammalian-sized genomes using the ultra high-throughput short read aligner Bowtie,
        and then analyzes the mapping results to identify splice junctions between exons.

        Args:
            index: Path to Bowtie index basename
            mate1: First mate FASTQ file
            mate2: Second mate FASTQ file (optional for paired-end)
            output_dir: Output directory for results
            threads: Number of threads to use
            library_type: Library type (fr-unstranded, fr-firststrand, fr-secondstrand)
            mate_inner_dist: Mean inner distance between mate pairs
            mate_std_dev: Standard deviation of inner distances
            min_anchor_length: Minimum anchor length for junctions
            splice_mismatches: Maximum mismatches allowed in junctions
            min_intron_length: Minimum intron length
            max_intron_length: Maximum intron length
            max_multihits: Maximum number of mappings allowed per read
            segment_mismatches: Maximum mismatches allowed in segment mapping
            segment_length: Length of segments for segment mapping
            no_novel_juncs: Only look for junctions in GTF file
            no_gtf_juncs: Don't use junctions from GTF file
            transcriptome_only: Only align to transcriptome
            microexon_search: Search for microexons

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        if not os.path.exists(mate1):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Mate 1 file does not exist: {mate1}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Mate 1 file not found: {mate1}",
            }

        if mate2 and not os.path.exists(mate2):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Mate 2 file does not exist: {mate2}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Mate 2 file not found: {mate2}",
            }

        # Build command
        cmd = ["tophat", "-o", output_dir, index, mate1]

        if mate2:
            cmd.append(mate2)

        # Add parameters
        cmd.extend(["-p", str(threads)])
        cmd.extend(["--library-type", library_type])
        cmd.extend(["--mate-inner-dist", str(mate_inner_dist)])
        cmd.extend(["--mate-std-dev", str(mate_std_dev)])
        cmd.extend(["--min-anchor-length", str(min_anchor_length)])
        cmd.extend(["--splice-mismatches", str(splice_mismatches)])
        cmd.extend(["--min-intron-length", str(min_intron_length)])
        cmd.extend(["--max-intron-length", str(max_intron_length)])
        cmd.extend(["--max-multihits", str(max_multihits)])
        cmd.extend(["--segment-mismatches", str(segment_mismatches)])
        cmd.extend(["--segment-length", str(segment_length)])

        if no_novel_juncs:
            cmd.append("--no-novel-juncs")
        if no_gtf_juncs:
            cmd.append("--no-gtf-juncs")
        if transcriptome_only:
            cmd.append("--transcriptome-only")
        if microexon_search:
            cmd.append("--microexon-search")

        try:
            # Execute TopHat alignment
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=output_dir
            )

            # Get output files
            output_files = []
            try:
                # TopHat creates several output files
                output_files = [
                    f"{output_dir}/accepted_hits.bam",
                    f"{output_dir}/junctions.bed",
                    f"{output_dir}/insertions.bed",
                    f"{output_dir}/deletions.bed",
                    f"{output_dir}/align_summary.txt",
                ]
                # Filter to only files that actually exist
                output_files = [f for f in output_files if os.path.exists(f)]
            except Exception:
                pass

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }

        except FileNotFoundError:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": "TopHat not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "TopHat not found in PATH",
            }
        except Exception as e:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Unexpected error: {e}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Unexpected error: {e}",
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy TopHat server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-tophat-server-{id(self)}")

            # Install TopHat
            container.with_command(
                "bash -c 'apt-get update && apt-get install -y tophat && tail -f /dev/null'"
            )

            # Start container
            container.start()

            # Wait for container to be ready
            container.reload()
            while container.status != "running":
                await asyncio.sleep(0.1)
                container.reload()

            # Store container info
            self.container_id = container.get_wrapped_container().id
            self.container_name = container.get_wrapped_container().name

            return MCPServerDeployment(
                server_name=self.name,
                server_type=self.server_type,
                container_id=self.container_id,
                container_name=self.container_name,
                status=MCPServerStatus.RUNNING,
                created_at=datetime.now(),
                started_at=datetime.now(),
                tools_available=self.list_tools(),
                configuration=self.config,
            )

        except Exception as e:
            return MCPServerDeployment(
                server_name=self.name,
                server_type=self.server_type,
                status=MCPServerStatus.FAILED,
                error_message=str(e),
                configuration=self.config,
            )

    async def stop_with_testcontainers(self) -> bool:
        """Stop TopHat server deployed with testcontainers."""
        try:
            if self.container_id:
                from testcontainers.core.container import DockerContainer

                container = DockerContainer(self.container_id)
                container.stop()

                self.container_id = None
                self.container_name = None

                return True
            return False
        except Exception:
            return False

    def get_server_info(self) -> dict[str, Any]:
        """Get information about this TopHat server."""
        return {
            "name": self.name,
            "type": "tophat",
            "version": "2.1.1",
            "description": "TopHat RNA-seq splice-aware aligner server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
