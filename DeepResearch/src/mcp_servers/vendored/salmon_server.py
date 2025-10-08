"""
Salmon MCP Server - Vendored BioinfoMCP server for RNA-seq quantification.

This module implements a strongly-typed MCP server for Salmon, a fast and accurate
tool for quantifying the expression of transcripts from RNA-seq data, using Pydantic AI
patterns and testcontainers deployment.
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


class SalmonServer(MCPServerBase):
    """MCP Server for Salmon RNA-seq quantification tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="salmon-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"SALMON_VERSION": "1.10.1"},
                capabilities=["rna_seq", "quantification", "transcript_expression"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="salmon_index",
            description="Build Salmon index from transcriptome FASTA file",
            inputs={
                "transcripts": "str",
                "index": "str",
                "kmer_len": "int",
                "threads": "int",
                "type": "str",
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
                    "description": "Build Salmon index from transcriptome",
                    "parameters": {
                        "transcripts": "/data/transcripts.fa",
                        "index": "/data/salmon_index",
                        "kmer_len": 31,
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def salmon_index(
        self,
        transcripts: str,
        index: str,
        kmer_len: int = 31,
        threads: int = 1,
        type: str = "quasi",
    ) -> dict[str, Any]:
        """
        Build Salmon index from transcriptome FASTA file.

        This tool creates a Salmon index which is required for fast and accurate
        quantification of RNA-seq data using Salmon's quasi-mapping algorithm.

        Args:
            transcripts: Path to transcriptome FASTA file
            index: Path to output index directory
            kmer_len: K-mer length for index building
            threads: Number of threads to use
            type: Index type (quasi, fmd)

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
        if not os.path.exists(transcripts):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Transcript file does not exist: {transcripts}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Transcript file not found: {transcripts}",
            }

        # Build command
        cmd = [
            "salmon",
            "index",
            "-t",
            transcripts,
            "-i",
            index,
            "-k",
            str(kmer_len),
            "--threads",
            str(threads),
            "--type",
            type,
        ]

        try:
            # Execute Salmon index building
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # Salmon creates index directory with various files
                if os.path.exists(index) and os.path.isdir(index):
                    index_files = os.listdir(index)
                    output_files = [os.path.join(index, f) for f in index_files]
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
                "stderr": "Salmon not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Salmon not found in PATH",
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
            name="salmon_quant",
            description="Quantify RNA-seq reads using Salmon",
            inputs={
                "index": "str",
                "lib_type": "str",
                "mates1": "list[str]",
                "mates2": "List[str | None]",
                "unmated_reads": "List[str | None]",
                "output": "str",
                "threads": "int",
                "validate_mappings": "bool",
                "seq_bias": "bool",
                "gc_bias": "bool",
                "pos_bias": "bool",
                "num_bootstraps": "int",
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
                    "description": "Quantify paired-end RNA-seq reads",
                    "parameters": {
                        "index": "/data/salmon_index",
                        "lib_type": "A",
                        "mates1": ["/data/sample1_R1.fastq"],
                        "mates2": ["/data/sample1_R2.fastq"],
                        "output": "/data/salmon_quant",
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def salmon_quant(
        self,
        index: str,
        lib_type: str,
        mates1: list[str | None] | None = None,
        mates2: list[str | None] | None = None,
        unmated_reads: list[str | None] | None = None,
        output: str = ".",
        threads: int = 1,
        validate_mappings: bool = False,
        seq_bias: bool = False,
        gc_bias: bool = False,
        pos_bias: bool = False,
        num_bootstraps: int = 0,
    ) -> dict[str, Any]:
        """
        Quantify RNA-seq reads using Salmon.

        This tool quantifies transcript expression from RNA-seq data using Salmon's
        fast and accurate quasi-mapping algorithm.

        Args:
            index: Path to Salmon index
            lib_type: Library type (A, ISF, ISR, etc.)
            mates1: List of mate 1 FASTQ files
            mates2: List of mate 2 FASTQ files (for paired-end)
            unmated_reads: List of unmated read FASTQ files
            output: Output directory
            threads: Number of threads to use
            validate_mappings: Validate mappings
            seq_bias: Correct for sequence-specific bias
            gc_bias: Correct for GC-content bias
            pos_bias: Correct for positional bias
            num_bootstraps: Number of bootstrap samples

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate index exists
        if not os.path.exists(index):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Index directory does not exist: {index}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Index directory not found: {index}",
            }

        # Validate input files exist
        all_reads = []
        if mates1:
            all_reads.extend(mates1)
        if mates2:
            all_reads.extend(mates2)
        if unmated_reads:
            all_reads.extend(unmated_reads)

        for read_file in all_reads:
            if not os.path.exists(read_file):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"Read file does not exist: {read_file}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"Read file not found: {read_file}",
                }

        # Build command
        cmd = [
            "salmon",
            "quant",
            "-i",
            index,
            "-l",
            lib_type,
            "-p",
            str(threads),
            "-o",
            output,
        ]

        # Add read files
        if mates1:
            cmd.extend(["-1"] + mates1)
        if mates2:
            cmd.extend(["-2"] + mates2)
        if unmated_reads:
            cmd.extend(["-r"] + unmated_reads)

        # Add options
        if validate_mappings:
            cmd.append("--validateMappings")
        if seq_bias:
            cmd.append("--seqBias")
        if gc_bias:
            cmd.append("--gcBias")
        if pos_bias:
            cmd.append("--posBias")
        if num_bootstraps > 0:
            cmd.extend(["--numBootstraps", str(num_bootstraps)])

        try:
            # Execute Salmon quantification
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # Salmon creates various output files
                possible_outputs = [
                    os.path.join(output, "quant.sf"),
                    os.path.join(output, "lib_format_counts.json"),
                    os.path.join(output, "logs", "salmon_quant.log"),
                ]
                for filepath in possible_outputs:
                    if os.path.exists(filepath):
                        output_files.append(filepath)
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
                "stderr": "Salmon not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Salmon not found in PATH",
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
        """Deploy Salmon server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-salmon-server-{id(self)}")

            # Install Salmon
            container.with_command("bash -c 'pip install salmon && tail -f /dev/null'")

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
        """Stop Salmon server deployed with testcontainers."""
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
        """Get information about this Salmon server."""
        return {
            "name": self.name,
            "type": "salmon",
            "version": "1.10.1",
            "description": "Salmon RNA-seq quantification server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
