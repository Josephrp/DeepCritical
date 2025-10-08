"""
StringTie MCP Server - Vendored BioinfoMCP server for transcript assembly.

This module implements a strongly-typed MCP server for StringTie, a fast and
highly efficient assembler of RNA-seq alignments into potential transcripts,
using Pydantic AI patterns and testcontainers deployment.
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


class StringTieServer(MCPServerBase):
    """MCP Server for StringTie transcript assembly tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="stringtie-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"STRINGTIE_VERSION": "2.2.1"},
                capabilities=["rna_seq", "transcript_assembly", "gene_annotation"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="stringtie_assemble",
            description="Assemble transcripts from RNA-seq alignments using StringTie",
            inputs={
                "input_bam": "str",
                "output_gtf": "str",
                "reference_gtf": "str | None",
                "threads": "int",
                "min_isoform_abundance": "float",
                "min_length": "int",
                "min_read_coverage": "float",
                "max_multiread_fraction": "float",
                "min_anchor_length": "int",
                "min_junction_coverage": "int",
                "trim_transcript": "bool",
                "disable_trimming": "bool",
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
                    "description": "Assemble transcripts from RNA-seq BAM file",
                    "parameters": {
                        "input_bam": "/data/aligned_reads.bam",
                        "output_gtf": "/data/transcripts.gtf",
                        "reference_gtf": "/data/genes.gtf",
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def stringtie_assemble(
        self,
        input_bam: str,
        output_gtf: str,
        reference_gtf: str | None = None,
        threads: int = 1,
        min_isoform_abundance: float = 0.1,
        min_length: int = 200,
        min_read_coverage: float = 2.5,
        max_multiread_fraction: float = 0.95,
        min_anchor_length: int = 10,
        min_junction_coverage: int = 1,
        trim_transcript: bool = False,
        disable_trimming: bool = False,
    ) -> dict[str, Any]:
        """
        Assemble transcripts from RNA-seq alignments using StringTie.

        This tool assembles transcripts from RNA-seq alignments and quantifies
        their expression levels, optionally using a reference annotation.

        Args:
            input_bam: Input BAM file with RNA-seq alignments
            output_gtf: Output GTF file with assembled transcripts
            reference_gtf: Reference GTF file for guided assembly
            threads: Number of threads to use
            min_isoform_abundance: Minimum isoform abundance
            min_length: Minimum assembled transcript length
            min_read_coverage: Minimum read coverage for junctions
            max_multiread_fraction: Maximum fraction of multiread alignments
            min_anchor_length: Minimum anchor length for junctions
            min_junction_coverage: Minimum junction coverage
            trim_transcript: Trim transcript ends
            disable_trimming: Disable trimming of transcript ends

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
        if not os.path.exists(input_bam):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Input BAM file does not exist: {input_bam}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Input BAM file not found: {input_bam}",
            }

        if reference_gtf and not os.path.exists(reference_gtf):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Reference GTF file does not exist: {reference_gtf}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Reference GTF file not found: {reference_gtf}",
            }

        # Build command
        cmd = [
            "stringtie",
            input_bam,
            "-o",
            output_gtf,
            "-p",
            str(threads),
            "-f",
            str(min_isoform_abundance),
            "-m",
            str(min_length),
            "-c",
            str(min_read_coverage),
            "-M",
            str(max_multiread_fraction),
            "-a",
            str(min_anchor_length),
            "-j",
            str(min_junction_coverage),
        ]

        if reference_gtf:
            cmd.extend(["-G", reference_gtf])
        if trim_transcript:
            cmd.append("-t")
        if disable_trimming:
            cmd.append("-T")

        try:
            # Execute StringTie assembly
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output_gtf):
                output_files = [output_gtf]

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
                "stderr": "StringTie not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "StringTie not found in PATH",
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

    @mcp_tool(
        MCPToolSpec(
            name="stringtie_merge",
            description="Merge transcript assemblies from multiple StringTie runs",
            inputs={
                "input_gtfs": "list[str]",
                "output_gtf": "str",
                "reference_gtf": "str | None",
                "min_tpm": "float",
                "min_isoform_abundance": "float",
                "min_length": "int",
                "keep_merged": "bool",
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
                    "description": "Merge multiple transcript assemblies",
                    "parameters": {
                        "input_gtfs": ["/data/sample1.gtf", "/data/sample2.gtf"],
                        "output_gtf": "/data/merged_transcripts.gtf",
                        "reference_gtf": "/data/genes.gtf",
                    },
                }
            ],
        )
    )
    def stringtie_merge(
        self,
        input_gtfs: list[str],
        output_gtf: str,
        reference_gtf: str | None = None,
        min_tpm: float = 0.0,
        min_isoform_abundance: float = 0.0,
        min_length: int = 0,
        keep_merged: bool = False,
    ) -> dict[str, Any]:
        """
        Merge transcript assemblies from multiple StringTie runs.

        This tool merges multiple transcript assemblies into a single non-redundant
        set of transcripts, useful for creating a comprehensive annotation.

        Args:
            input_gtfs: List of input GTF files from StringTie
            output_gtf: Output merged GTF file
            reference_gtf: Reference GTF file for comparison
            min_tpm: Minimum TPM for transcript inclusion
            min_isoform_abundance: Minimum isoform abundance
            min_length: Minimum transcript length
            keep_merged: Keep merged transcripts

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        for input_gtf in input_gtfs:
            if not os.path.exists(input_gtf):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"Input GTF file does not exist: {input_gtf}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"Input GTF file not found: {input_gtf}",
                }

        if reference_gtf and not os.path.exists(reference_gtf):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Reference GTF file does not exist: {reference_gtf}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Reference GTF file not found: {reference_gtf}",
            }

        # Create temporary merge list file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for gtf in input_gtfs:
                f.write(f"{gtf}\n")
            merge_list = f.name

        try:
            # Build command
            cmd = [
                "stringtie",
                "--merge",
                "-o",
                output_gtf,
                merge_list,
            ]

            if reference_gtf:
                cmd.extend(["-G", reference_gtf])
            if min_tpm > 0:
                cmd.extend(["-T", str(min_tpm)])
            if min_isoform_abundance > 0:
                cmd.extend(["-f", str(min_isoform_abundance)])
            if min_length > 0:
                cmd.extend(["-m", str(min_length)])
            if keep_merged:
                cmd.append("-F")

            # Execute StringTie merge
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output_gtf):
                output_files = [output_gtf]

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
                "stderr": "StringTie not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "StringTie not found in PATH",
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
        finally:
            # Clean up temporary file
            try:
                os.unlink(merge_list)
            except Exception:
                pass

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy StringTie server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-stringtie-server-{id(self)}")

            # Install StringTie
            container.with_command(
                "bash -c 'pip install stringtie && tail -f /dev/null'"
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
        """Stop StringTie server deployed with testcontainers."""
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
        """Get information about this StringTie server."""
        return {
            "name": self.name,
            "type": "stringtie",
            "version": "2.2.1",
            "description": "StringTie transcript assembly server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
