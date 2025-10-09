"""
BCFtools MCP Server - Vendored BioinfoMCP server for BCF/VCF file operations.

This module implements a strongly-typed MCP server for BCFtools, a suite of programs
for manipulating variant calls in the Variant Call Format (VCF) and its binary
counterpart BCF.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
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


class BCFtoolsServer(MCPServerBase):
    """MCP Server for BCFtools variant analysis utilities."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="bcftools-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"BCFTOOLS_VERSION": "1.17"},
                capabilities=["variant_analysis", "vcf_processing", "genomics"],
            )
        super().__init__(config)

    @mcp_tool()
    def bcftools_view(
        self,
        input_file: str,
        output_file: str | None = None,
        output_type: str = "v",
        regions: str | None = None,
        targets: str | None = None,
        samples: str | None = None,
        include: str | None = None,
        exclude: str | None = None,
        apply_filters: str | None = None,
        threads: int = 0,
        header_only: bool = False,
        no_header: bool = False,
        compression_level: int = -1,
    ) -> dict[str, Any]:
        """
        View, subset and filter VCF/BCF files.

        Args:
            input_file: Input VCF/BCF file
            output_file: Output file (optional, stdout if not specified)
            output_type: Output format: v=VCF, b=BCF, u=uncompressed BCF, z=compressed VCF
            regions: Restrict to comma-separated list of regions
            targets: Similar to -r but streams rather than index-jumps
            samples: List of samples to include
            include: Include sites for which expression is true
            exclude: Exclude sites for which expression is true
            apply_filters: Require at least one of the listed FILTER strings
            threads: Number of threads to use
            header_only: Output only the header
            no_header: Suppress the header in VCF output
            compression_level: Compression level (0-9, -1=system default)

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Build command
        cmd = ["bcftools", "view"]

        # Add options
        if output_type:
            cmd.extend(["-O", output_type])
        if regions:
            cmd.extend(["-r", regions])
        if targets:
            cmd.extend(["-t", targets])
        if samples:
            cmd.extend(["-s", samples])
        if include:
            cmd.extend(["-i", include])
        if exclude:
            cmd.extend(["-e", exclude])
        if apply_filters:
            cmd.extend(["-f", apply_filters])
        if threads > 0:
            cmd.extend(["--threads", str(threads)])
        if header_only:
            cmd.append("-h")
        if no_header:
            cmd.append("-H")
        if compression_level >= 0:
            cmd.extend(["-l", str(compression_level)])

        # Add input file
        cmd.append(input_file)

        # Execute command
        try:
            if output_file:
                # Redirect output to file
                with open(output_file, "w") as f:
                    result = subprocess.run(
                        cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True
                    )
                stdout = ""
                stderr = result.stderr
                output_files = [output_file]
            else:
                # Capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                stdout = result.stdout
                stderr = result.stderr
                output_files = []

            return {
                "command_executed": " ".join(cmd),
                "stdout": stdout,
                "stderr": stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout if e.stdout else "",
                "stderr": e.stderr if e.stderr else "",
                "output_files": [],
                "exit_code": e.returncode,
                "success": False,
                "error": f"bcftools view execution failed: {e}",
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
    def bcftools_stats(
        self,
        input_file: str,
        output_file: str | None = None,
        samples: str | None = None,
        regions: str | None = None,
        targets: str | None = None,
        apply_filters: str | None = None,
        threads: int = 0,
    ) -> dict[str, Any]:
        """
        Parse VCF/BCF files and generate statistics.

        Args:
            input_file: Input VCF/BCF file
            output_file: Output file (optional, stdout if not specified)
            samples: List of samples to include
            regions: Restrict to comma-separated list of regions
            targets: Similar to -r but streams rather than index-jumps
            apply_filters: Require at least one of the listed FILTER strings
            threads: Number of threads to use

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Build command
        cmd = ["bcftools", "stats"]

        # Add options
        if samples:
            cmd.extend(["-s", samples])
        if regions:
            cmd.extend(["-r", regions])
        if targets:
            cmd.extend(["-t", targets])
        if apply_filters:
            cmd.extend(["-f", apply_filters])
        if threads > 0:
            cmd.extend(["--threads", str(threads)])

        # Add input file
        cmd.append(input_file)

        # Execute command
        try:
            if output_file:
                # Redirect output to file
                with open(output_file, "w") as f:
                    result = subprocess.run(
                        cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True
                    )
                stdout = ""
                stderr = result.stderr
                output_files = [output_file]
            else:
                # Capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                stdout = result.stdout
                stderr = result.stderr
                output_files = []

            return {
                "command_executed": " ".join(cmd),
                "stdout": stdout,
                "stderr": stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout if e.stdout else "",
                "stderr": e.stderr if e.stderr else "",
                "output_files": [],
                "exit_code": e.returncode,
                "success": False,
                "error": f"bcftools stats execution failed: {e}",
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
    def bcftools_filter(
        self,
        input_file: str,
        output_file: str | None = None,
        include: str | None = None,
        exclude: str | None = None,
        soft_filter: str | None = None,
        mode: str | None = None,
        regions: str | None = None,
        targets: str | None = None,
        samples: str | None = None,
        threads: int = 0,
    ) -> dict[str, Any]:
        """
        Filter VCF/BCF files using arbitrary expressions.

        Args:
            input_file: Input VCF/BCF file
            output_file: Output file (optional, stdout if not specified)
            include: Include sites for which expression is true
            exclude: Exclude sites for which expression is true
            soft_filter: Apply soft filter
            mode: Mode of filtering: +, x, =
            regions: Restrict to comma-separated list of regions
            targets: Similar to -r but streams rather than index-jumps
            samples: List of samples to include
            threads: Number of threads to use

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Build command
        cmd = ["bcftools", "filter"]

        # Add options
        if include:
            cmd.extend(["-i", include])
        if exclude:
            cmd.extend(["-e", exclude])
        if soft_filter:
            cmd.extend(["-s", soft_filter])
        if mode:
            cmd.extend(["-m", mode])
        if regions:
            cmd.extend(["-r", regions])
        if targets:
            cmd.extend(["-t", targets])
        if samples:
            cmd.extend(["-s", samples])
        if threads > 0:
            cmd.extend(["--threads", str(threads)])

        # Add input file
        cmd.append(input_file)

        # Execute command
        try:
            if output_file:
                # Redirect output to file
                with open(output_file, "w") as f:
                    result = subprocess.run(
                        cmd, stdout=f, stderr=subprocess.PIPE, text=True, check=True
                    )
                stdout = ""
                stderr = result.stderr
                output_files = [output_file]
            else:
                # Capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                stdout = result.stdout
                stderr = result.stderr
                output_files = []

            return {
                "command_executed": " ".join(cmd),
                "stdout": stdout,
                "stderr": stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout if e.stdout else "",
                "stderr": e.stderr if e.stderr else "",
                "output_files": [],
                "exit_code": e.returncode,
                "success": False,
                "error": f"bcftools filter execution failed: {e}",
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

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy the BCFtools server using testcontainers."""
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
        """Stop the BCFtools server deployed with testcontainers."""
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
        """Get information about this BCFtools server."""
        return {
            "name": self.name,
            "type": self.server_type.value,
            "version": "1.17",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
            "capabilities": self.config.capabilities,
            "pydantic_ai_enabled": self.pydantic_ai_agent is not None,
            "session_active": self.session is not None,
        }


# Create server instance
bcftools_server = BCFtoolsServer()
