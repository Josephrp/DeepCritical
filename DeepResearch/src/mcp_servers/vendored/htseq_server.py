"""
HTSeq MCP Server - Vendored BioinfoMCP server for read counting.

This module implements a strongly-typed MCP server for HTSeq, a tool for
analyzing high-throughput sequencing data with a focus on RNA-seq, using
Pydantic AI patterns and testcontainers deployment.
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


class HTSeqServer(MCPServerBase):
    """MCP Server for HTSeq read counting tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="htseq-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"HTSEQ_VERSION": "2.0.5"},
                capabilities=["rna_seq", "read_counting", "gene_expression"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="htseq_count",
            description="Count reads overlapping genomic features using HTSeq",
            inputs={
                "sam_file": "str",
                "gtf_file": "str",
                "output_file": "str",
                "format": "str",
                "stranded": "str",
                "mode": "str",
                "feature_type": "str",
                "idattr": "str",
                "minaqual": "int",
                "secondary_alignments": "str",
                "supplementary_alignments": "str",
                "order": "str",
                "max_buffer_size_mb": "int",
                "quiet": "bool",
                "additional_attr": "str",
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
                    "description": "Count reads overlapping genes in SAM file",
                    "parameters": {
                        "sam_file": "/data/aligned_reads.sam",
                        "gtf_file": "/data/genes.gtf",
                        "output_file": "/data/counts.txt",
                        "format": "sam",
                        "stranded": "no",
                        "mode": "union",
                        "feature_type": "exon",
                        "idattr": "gene_id",
                    },
                }
            ],
        )
    )
    def htseq_count(
        self,
        sam_file: str,
        gtf_file: str,
        output_file: str,
        format: str = "sam",
        stranded: str = "no",
        mode: str = "union",
        feature_type: str = "exon",
        idattr: str = "gene_id",
        minaqual: int = 10,
        secondary_alignments: str = "ignore",
        supplementary_alignments: str = "ignore",
        order: str = "name",
        max_buffer_size_mb: int = 300,
        quiet: bool = False,
        additional_attr: str = "",
    ) -> dict[str, Any]:
        """
        Count reads overlapping genomic features using HTSeq.

        This tool counts reads that overlap with genomic features such as genes,
        producing a count matrix for downstream differential expression analysis.

        Args:
            sam_file: Input SAM/BAM file with aligned reads
            gtf_file: GTF/GFF annotation file
            output_file: Output count file
            format: Input file format (sam, bam)
            stranded: Strandedness of the library (no, yes, reverse)
            mode: Counting mode (union, intersection-strict, intersection-nonempty)
            feature_type: Feature type to count (exon, gene, etc.)
            idattr: Attribute to use as feature ID (gene_id, transcript_id, etc.)
            minaqual: Minimum alignment quality
            secondary_alignments: How to handle secondary alignments (ignore, score)
            supplementary_alignments: How to handle supplementary alignments (ignore, score)
            order: Read sorting order (name, pos)
            max_buffer_size_mb: Maximum buffer size in MB
            quiet: Suppress progress messages
            additional_attr: Additional attributes to include

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        if not os.path.exists(sam_file):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Input SAM/BAM file does not exist: {sam_file}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Input file not found: {sam_file}",
            }

        if not os.path.exists(gtf_file):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"GTF file does not exist: {gtf_file}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"GTF file not found: {gtf_file}",
            }

        # Build command
        cmd = [
            "htseq-count",
            "-f",
            format,
            "-r",
            order,
            "-s",
            stranded,
            "-a",
            str(minaqual),
            "-t",
            feature_type,
            "-i",
            idattr,
            "-m",
            mode,
        ]

        # Add options
        if secondary_alignments != "ignore":
            cmd.extend(["--secondary-alignments", secondary_alignments])
        if supplementary_alignments != "ignore":
            cmd.extend(["--supplementary-alignments", supplementary_alignments])
        if max_buffer_size_mb != 300:
            cmd.extend(["--max-buffer-size", f"{max_buffer_size_mb}M"])
        if quiet:
            cmd.append("--quiet")
        if additional_attr:
            cmd.extend(["--additional-attr", additional_attr])

        # Add input files
        cmd.extend([sam_file, gtf_file])

        try:
            # Execute HTSeq counting with output redirection
            with open(output_file, "w") as outfile:
                result = subprocess.run(
                    cmd,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

            # Get output files
            output_files = []
            if os.path.exists(output_file):
                output_files = [output_file]

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
                "stderr": "HTSeq not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "HTSeq not found in PATH",
            }
        except Exception as e:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": str(e),
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": str(e),
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy HTSeq server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-htseq-server-{id(self)}")

            # Install HTSeq
            container.with_command("bash -c 'pip install HTSeq && tail -f /dev/null'")

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
        """Stop HTSeq server deployed with testcontainers."""
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
        """Get information about this HTSeq server."""
        return {
            "name": self.name,
            "type": "htseq",
            "version": "2.0.5",
            "description": "HTSeq read counting server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
