"""
Cutadapt MCP Server - Vendored BioinfoMCP server for adapter trimming.

This module implements a strongly-typed MCP server for Cutadapt, a tool for
trimming adapters from high-throughput sequencing reads, using Pydantic AI
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

from ...datatypes.bioinformatics_mcp import MCPServerBase, mcp_tool
from ...datatypes.mcp import (
    MCPAgentIntegration,
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)


class CutadaptServer(MCPServerBase):
    """MCP Server for Cutadapt adapter trimming tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="cutadapt-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"CUTADAPT_VERSION": "4.4"},
                capabilities=[
                    "adapter_trimming",
                    "quality_filtering",
                    "read_preprocessing",
                ],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="cutadapt_trim",
            description="Trim adapters and low-quality bases from FASTQ files using Cutadapt",
            inputs={
                "input_files": "list[str]",
                "output_files": "list[str]",
                "adapter_5prime": "str | None",
                "adapter_3prime": "str | None",
                "adapter_anywhere": "str | None",
                "front": "str | None",
                "anywhere": "str | None",
                "quality_cutoff": "int",
                "minimum_length": "int",
                "maximum_length": "int",
                "max_n": "float",
                "trim_n": "bool",
                "length": "int | None",
                "prefix": "str | None",
                "suffix": "str | None",
                "times": "int",
                "overlap": "int",
                "error_rate": "float",
                "no_indels": "bool",
                "match_read_wildcards": "bool",
                "no_match_adapter_wildcards": "bool",
                "action": "str",
                "discard_casava": "bool",
                "discard_trimmed": "bool",
                "discard_untrimmed": "bool",
                "discard_cassava": "bool",
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
                    "description": "Trim Illumina adapters from paired-end reads",
                    "parameters": {
                        "input_files": [
                            "/data/sample_R1.fastq.gz",
                            "/data/sample_R2.fastq.gz",
                        ],
                        "output_files": [
                            "/data/sample_R1_trimmed.fastq.gz",
                            "/data/sample_R2_trimmed.fastq.gz",
                        ],
                        "adapter_3prime": "AGATCGGAAGAG",
                        "quality_cutoff": 20,
                        "minimum_length": 20,
                    },
                }
            ],
        )
    )
    def cutadapt_trim(
        self,
        input_files: list[str],
        output_files: list[str],
        adapter_5prime: str | None = None,
        adapter_3prime: str | None = None,
        adapter_anywhere: str | None = None,
        front: str | None = None,
        anywhere: str | None = None,
        quality_cutoff: int = 20,
        minimum_length: int = 20,
        maximum_length: int = 0,
        max_n: float = 0.0,
        trim_n: bool = False,
        length: int | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        times: int = 1,
        overlap: int = 3,
        error_rate: float = 0.1,
        no_indels: bool = False,
        match_read_wildcards: bool = False,
        no_match_adapter_wildcards: bool = False,
        action: str = "trim",
        discard_casava: bool = False,
        discard_trimmed: bool = False,
        discard_untrimmed: bool = False,
        discard_cassava: bool = False,
    ) -> dict[str, Any]:
        """
        Trim adapters and low-quality bases from FASTQ files using Cutadapt.

        This tool removes adapters and low-quality bases from high-throughput sequencing reads.

        Args:
            input_files: Input FASTQ files
            output_files: Output FASTQ files (must match input_files length for paired-end)
            adapter_5prime: 5' adapter sequence
            adapter_3prime: 3' adapter sequence
            adapter_anywhere: Adapter that may appear anywhere
            front: Sequence to remove from the beginning
            anywhere: Sequence to remove from anywhere in the read
            quality_cutoff: Quality cutoff for trimming (3' end)
            minimum_length: Minimum length of reads to keep
            maximum_length: Maximum length of reads to keep
            max_n: Maximum fraction of Ns allowed in a read
            trim_n: Trim Ns from reads
            length: Truncate reads to this length
            prefix: Add prefix to read names
            suffix: Add suffix to read names
            times: How many times to try to remove adapters
            overlap: Minimum overlap between adapter and read
            error_rate: Maximum error rate for adapter matching
            no_indels: Do not allow insertions or deletions in adapter matching
            match_read_wildcards: Allow wildcards in reads
            no_match_adapter_wildcards: Do not allow wildcards in adapters
            action: What to do when adapter is found
            discard_casava: Discard reads that fail Casava chastity check
            discard_trimmed: Discard reads that were trimmed
            discard_untrimmed: Discard reads that were not trimmed
            discard_cassava: Discard reads that fail Casava filter

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        for input_file in input_files:
            if not os.path.exists(input_file):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"Input file does not exist: {input_file}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"Input file not found: {input_file}",
                }

        # Build command
        cmd = ["cutadapt"]

        # Add adapters
        if adapter_5prime:
            cmd.extend(["-g", adapter_5prime])
        if adapter_3prime:
            cmd.extend(["-a", adapter_3prime])
        if adapter_anywhere:
            cmd.extend(["-b", adapter_anywhere])
        if front:
            cmd.extend(["-G", front])  # For paired-end reads
        if anywhere:
            cmd.extend(["-B", anywhere])  # For paired-end reads

        # Add quality and length parameters
        cmd.extend(["-q", str(quality_cutoff)])
        cmd.extend(["-m", str(minimum_length)])
        if maximum_length > 0:
            cmd.extend(["-M", str(maximum_length)])
        if max_n > 0.0:
            cmd.extend(["--max-n", str(max_n)])
        if trim_n:
            cmd.append("--trim-n")
        if length:
            cmd.extend(["-l", str(length)])
        if prefix:
            cmd.extend(["--prefix", prefix])
        if suffix:
            cmd.extend(["--suffix", suffix])

        # Add adapter matching parameters
        cmd.extend(["-n", str(times)])
        cmd.extend(["-O", str(overlap)])
        cmd.extend(["-e", str(error_rate)])
        if no_indels:
            cmd.append("--no-indels")
        if match_read_wildcards:
            cmd.append("--match-read-wildcards")
        if no_match_adapter_wildcards:
            cmd.append("--no-match-adapter-wildcards")
        cmd.extend(["--action", action])

        # Add filtering options
        if discard_casava:
            cmd.append("--discard-casava")
        if discard_trimmed:
            cmd.append("--discard-trimmed")
        if discard_untrimmed:
            cmd.append("--discard-untrimmed")
        if discard_cassava:
            cmd.append("--discard-cassava")

        # Add input and output files
        if len(input_files) == 2 and len(output_files) == 2:
            # Paired-end mode
            cmd.extend(["-o", output_files[0], "-p", output_files[1]])
            cmd.extend(input_files)
        else:
            # Single-end mode
            for i, input_file in enumerate(input_files):
                output_file = (
                    output_files[i]
                    if i < len(output_files)
                    else input_file.replace(".fastq", "_trimmed.fastq")
                )
                cmd.extend(["-o", output_file, input_file])

        try:
            # Execute Cutadapt
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            final_output_files = []
            if len(input_files) == 2 and len(output_files) == 2:
                final_output_files = output_files
            else:
                for i, input_file in enumerate(input_files):
                    output_file = (
                        output_files[i]
                        if i < len(output_files)
                        else input_file.replace(".fastq", "_trimmed.fastq")
                    )
                    if os.path.exists(output_file):
                        final_output_files.append(output_file)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": final_output_files,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }

        except FileNotFoundError:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": "Cutadapt not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Cutadapt not found in PATH",
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
        """Deploy Cutadapt server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-cutadapt-server-{id(self)}")

            # Install Cutadapt
            container.with_command(
                "bash -c 'pip install cutadapt && tail -f /dev/null'"
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
        """Stop Cutadapt server deployed with testcontainers."""
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
        """Get information about this Cutadapt server."""
        return {
            "name": self.name,
            "type": "cutadapt",
            "version": "4.4",
            "description": "Cutadapt adapter trimming server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
