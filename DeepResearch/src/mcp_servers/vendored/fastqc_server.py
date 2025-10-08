"""
FastQC MCP Server - Vendored BioinfoMCP server for quality control of FASTQ files.

This module implements a strongly-typed MCP server for FastQC, a popular tool
for quality control checks on high throughput sequence data, using Pydantic AI patterns
and testcontainers deployment.
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


class FastQCServer(MCPServerBase):
    """MCP Server for FastQC quality control tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="fastqc-server",
                server_type=MCPServerType.FASTQC,
                container_image="python:3.11-slim",
                environment_variables={"FASTQC_VERSION": "0.11.9"},
                capabilities=["quality_control", "sequence_analysis", "fastq"],
            )
        super().__init__(config)

    @mcp_tool()
    def run_fastqc(
        self,
        input_files: list[str],
        output_dir: str,
        extract: bool = False,
        format: str = "fastq",
        contaminants: str | None = None,
        adapters: str | None = None,
        limits: str | None = None,
        kmers: int = 7,
        threads: int = 1,
        quiet: bool = False,
        nogroup: bool = False,
        min_length: int = 0,
        max_length: int = 0,
        casava: bool = False,
        nano: bool = False,
        nofilter: bool = False,
        outdir: str | None = None,
    ) -> dict[str, Any]:
        """
        Run FastQC quality control on input FASTQ files.

        Args:
            input_files: List of input FASTQ files to analyze
            output_dir: Output directory for results
            extract: Extract compressed files
            format: Input file format (fastq, bam, sam)
            contaminants: File containing contaminants to screen for
            adapters: File containing adapter sequences
            limits: File containing analysis limits
            kmers: Length of Kmer to look for
            threads: Number of threads to use
            quiet: Suppress progress messages
            nogroup: Disable grouping of bases for reads >50bp
            min_length: Minimum sequence length to include
            max_length: Maximum sequence length to include
            casava: Expect CASAVA format files
            nano: Expect NanoPore/ONT data
            nofilter: Do not filter out low quality sequences
            outdir: Alternative output directory (overrides output_dir)

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate input files
        if not input_files:
            raise ValueError("At least one input file must be specified")

        # Validate input files exist
        for input_file in input_files:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

        # Use alternative output directory if specified
        if outdir:
            output_dir = outdir

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = ["fastqc"]

        # Add options
        if extract:
            cmd.append("--extract")
        if format != "fastq":
            cmd.extend(["--format", format])
        if contaminants:
            cmd.extend(["--contaminants", contaminants])
        if adapters:
            cmd.extend(["--adapters", adapters])
        if limits:
            cmd.extend(["--limits", limits])
        if kmers != 7:
            cmd.extend(["--kmers", str(kmers)])
        if threads != 1:
            cmd.extend(["--threads", str(threads)])
        if quiet:
            cmd.append("--quiet")
        if nogroup:
            cmd.append("--nogroup")
        if min_length > 0:
            cmd.extend(["--min_length", str(min_length)])
        if max_length > 0:
            cmd.extend(["--max_length", str(max_length)])
        if casava:
            cmd.append("--casava")
        if nano:
            cmd.append("--nano")
        if nofilter:
            cmd.append("--nofilter")

        # Add input files
        cmd.extend(input_files)

        # Execute command
        try:
            result = subprocess.run(
                cmd, cwd=output_dir, capture_output=True, text=True, check=True
            )

            # Find output files
            output_files = []
            for input_file in input_files:
                # Get base name without extension
                base_name = Path(input_file).stem
                if base_name.endswith(".fastq") or base_name.endswith(".fq"):
                    base_name = Path(base_name).stem

                # Look for HTML and ZIP files
                html_file = Path(output_dir) / f"{base_name}_fastqc.html"
                zip_file = Path(output_dir) / f"{base_name}_fastqc.zip"

                if html_file.exists():
                    output_files.append(str(html_file))
                if zip_file.exists():
                    output_files.append(str(zip_file))

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "exit_code": e.returncode,
                "success": False,
                "error": f"FastQC execution failed: {e}",
            }

        except Exception as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": str(e),
            }

    @mcp_tool()
    def check_fastqc_version(self) -> dict[str, Any]:
        """Check the version of FastQC installed."""
        cmd = ["fastqc", "--version"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout.strip(),
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "success": True,
                "version": result.stdout.strip(),
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "exit_code": e.returncode,
                "success": False,
                "error": f"Failed to check FastQC version: {e}",
            }

        except FileNotFoundError:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "success": False,
                "error": "FastQC not found in PATH",
            }

    @mcp_tool()
    def list_fastqc_outputs(self, output_dir: str) -> dict[str, Any]:
        """List FastQC output files in the specified directory."""
        try:
            path = Path(output_dir)

            if not path.exists():
                return {
                    "command_executed": f"list_fastqc_outputs {output_dir}",
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1,
                    "success": False,
                    "error": f"Output directory does not exist: {output_dir}",
                }

            # Find FastQC output files
            html_files = list(path.glob("*_fastqc.html"))

            files = []
            for html_file in html_files:
                zip_file = html_file.with_suffix(".zip")
                files.append(
                    {
                        "html_file": str(html_file),
                        "zip_file": str(zip_file) if zip_file.exists() else None,
                        "base_name": html_file.stem.replace("_fastqc", ""),
                    }
                )

            return {
                "command_executed": f"list_fastqc_outputs {output_dir}",
                "stdout": f"Found {len(files)} FastQC output file(s)",
                "stderr": "",
                "exit_code": 0,
                "success": True,
                "files": files,
                "output_directory": str(path),
            }

        except Exception as e:
            return {
                "command_executed": f"list_fastqc_outputs {output_dir}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "success": False,
                "error": f"Failed to list FastQC outputs: {e}",
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy the FastQC server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer
            from testcontainers.core.waiting_utils import wait_for_logs

            # Create container
            container_name = f"mcp-{self.name}-{id(self)}"
            container = DockerContainer(self.config.container_image)
            container.with_name(container_name)

            # Set environment variables
            for key, value in self.config.environment_variables.items():
                container.with_env(key, value)

            # Add volume for data exchange
            container.with_volume_mapping("/tmp", "/tmp")

            # Set resource limits
            if self.config.resource_limits.memory:
                # Note: testcontainers doesn't directly support memory limits
                pass

            if self.config.resource_limits.cpu:
                # Note: testcontainers doesn't directly support CPU limits
                pass

            # Start container
            container.start()

            # Wait for container to be ready
            wait_for_logs(container, "Python", timeout=30)

            # Update deployment info
            deployment = MCPServerDeployment(
                server_name=self.name,
                server_type=self.server_type,
                container_id=container.get_wrapped_container().id,
                container_name=container_name,
                status=MCPServerStatus.RUNNING,
                created_at=datetime.now(),
                started_at=datetime.now(),
                tools_available=self.list_tools(),
                configuration=self.config,
            )

            self.container_id = container.get_wrapped_container().id
            self.container_name = container_name

            return deployment

        except Exception as e:
            return MCPServerDeployment(
                server_name=self.name,
                server_type=self.server_type,
                status=MCPServerStatus.FAILED,
                error_message=str(e),
                configuration=self.config,
            )

    async def stop_with_testcontainers(self) -> bool:
        """Stop the FastQC server deployed with testcontainers."""
        if not self.container_id:
            return False

        try:
            from testcontainers.core.container import DockerContainer

            container = DockerContainer(self.container_id)
            container.stop()

            self.container_id = None
            self.container_name = None

            return True

        except Exception as e:
            self.logger.error(f"Failed to stop container {self.container_id}: {e}")
            return False

    def get_server_info(self) -> dict[str, Any]:
        """Get information about this FastQC server."""
        return {
            "name": self.name,
            "type": self.server_type.value,
            "version": "0.11.9",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
            "capabilities": self.config.capabilities,
        }


# Create server instance
fastqc_server = FastQCServer()
