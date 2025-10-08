"""
BEDtools MCP Server - Vendored BioinfoMCP server for BED file operations.

This module implements a strongly-typed MCP server for BEDtools, a suite of utilities
for comparing, summarizing, and intersecting genomic features in BED format.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
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


class BEDToolsServer(MCPServerBase):
    """MCP Server for BEDtools genomic arithmetic utilities."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="bedtools-server",
                server_type=MCPServerType.BEDTOOLS,
                container_image="python:3.11-slim",
                environment_variables={"BEDTOOLS_VERSION": "2.30.0"},
                capabilities=["genomics", "bed_operations", "interval_arithmetic"],
            )
        super().__init__(config)

    @mcp_tool()
    def bedtools_intersect(
        self,
        a_file: str,
        b_files: list[str],
        output_file: str | None = None,
        wa: bool = False,
        wb: bool = False,
        loj: bool = False,
        wo: bool = False,
        wao: bool = False,
        u: bool = False,
        c: bool = False,
        v: bool = False,
        f: float = 1e-9,
        fraction_b: float = 1e-9,
        r: bool = False,
        e: bool = False,
        s: bool = False,
        sorted_input: bool = False,
    ) -> dict[str, Any]:
        """
        Find overlapping intervals between two sets of genomic features.

        Args:
            a_file: Path to file A (BED/GFF/VCF)
            b_files: List of files B (BED/GFF/VCF)
            output_file: Output file (optional, stdout if not specified)
            wa: Write original entry in A for each overlap
            wb: Write original entry in B for each overlap
            loj: Left outer join; report all A features with or without overlaps
            wo: Write original A and B entries plus number of base pairs of overlap
            wao: Like -wo but also report A features without overlap with overlap=0
            u: Write original A entry once if any overlaps found in B
            c: For each A entry, report number of hits in B
            v: Only report A entries with no overlap in B
            f: Minimum overlap fraction of A (0.0-1.0)
            fraction_b: Minimum overlap fraction of B (0.0-1.0)
            r: Require reciprocal overlap fraction for A and B
            e: Require minimum fraction satisfied for A OR B
            s: Force strandedness (overlaps on same strand only)
            sorted_input: Use memory-efficient algorithm for sorted input

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate input files
        if not os.path.exists(a_file):
            raise FileNotFoundError(f"Input file A not found: {a_file}")

        for b_file in b_files:
            if not os.path.exists(b_file):
                raise FileNotFoundError(f"Input file B not found: {b_file}")

        # Validate parameters
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"Parameter f must be between 0.0 and 1.0, got {f}")
        if not (0.0 <= fraction_b <= 1.0):
            raise ValueError(
                f"Parameter fraction_b must be between 0.0 and 1.0, got {fraction_b}"
            )

        # Build command
        cmd = ["bedtools", "intersect"]

        # Add options
        if wa:
            cmd.append("-wa")
        if wb:
            cmd.append("-wb")
        if loj:
            cmd.append("-loj")
        if wo:
            cmd.append("-wo")
        if wao:
            cmd.append("-wao")
        if u:
            cmd.append("-u")
        if c:
            cmd.append("-c")
        if v:
            cmd.append("-v")
        if f != 1e-9:
            cmd.extend(["-f", str(f)])
        if fraction_b != 1e-9:
            cmd.extend(["-F", str(fraction_b)])
        if r:
            cmd.append("-r")
        if e:
            cmd.append("-e")
        if s:
            cmd.append("-s")
        if sorted_input:
            cmd.append("-sorted")

        # Add input files
        cmd.extend(["-a", a_file])
        for b_file in b_files:
            cmd.extend(["-b", b_file])

        # Execute command
        try:
            if output_file:
                # Redirect output to file
                with open(output_file, "w") as output_handle:
                    result = subprocess.run(
                        cmd,
                        stdout=output_handle,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
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

        except subprocess.CalledProcessError as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": exc.stdout if exc.stdout else "",
                "stderr": exc.stderr if exc.stderr else "",
                "output_files": [],
                "exit_code": exc.returncode,
                "success": False,
                "error": f"bedtools intersect execution failed: {exc}",
            }

        except Exception as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": str(exc),
            }

    @mcp_tool()
    def bedtools_merge(
        self,
        input_file: str,
        output_file: str | None = None,
        d: int = 0,
        c: list[str] | None = None,
        o: list[str] | None = None,
        delim: str = ",",
        s: bool = False,
        strand_filter: str | None = None,
        header: bool = False,
    ) -> dict[str, Any]:
        """
        Merge overlapping/adjacent intervals.

        Args:
            input_file: Input BED file
            output_file: Output file (optional, stdout if not specified)
            d: Maximum distance between features allowed for merging
            c: Columns from input file to operate upon
            o: Operations to perform on specified columns
            delim: Delimiter for merged columns
            s: Force merge within same strand
            strand_filter: Only merge intervals with matching strand
            header: Print header

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Build command
        cmd = ["bedtools", "merge"]

        # Add options
        if d > 0:
            cmd.extend(["-d", str(d)])
        if c:
            cmd.extend(["-c", ",".join(c)])
        if o:
            cmd.extend(["-o", ",".join(o)])
        if delim != ",":
            cmd.extend(["-delim", delim])
        if s:
            cmd.append("-s")
        if strand_filter:
            cmd.extend(["-S", strand_filter])
        if header:
            cmd.append("-header")

        # Add input file
        cmd.extend(["-i", input_file])

        # Execute command
        try:
            if output_file:
                # Redirect output to file
                with open(output_file, "w") as output_handle:
                    result = subprocess.run(
                        cmd,
                        stdout=output_handle,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
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

        except subprocess.CalledProcessError as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": exc.stdout if exc.stdout else "",
                "stderr": exc.stderr if exc.stderr else "",
                "output_files": [],
                "exit_code": exc.returncode,
                "success": False,
                "error": f"bedtools merge execution failed: {exc}",
            }

        except Exception as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": str(exc),
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy the BEDtools server using testcontainers."""
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

        except Exception as deploy_exc:
            return MCPServerDeployment(
                server_name=self.name,
                server_type=self.server_type,
                status=MCPServerStatus.FAILED,
                error_message=str(deploy_exc),
                configuration=self.config,
            )

    async def stop_with_testcontainers(self) -> bool:
        """Stop the BEDtools server deployed with testcontainers."""
        if not self.container_id:
            return False

        try:
            from testcontainers.core.container import DockerContainer

            container = DockerContainer(self.container_id)
            container.stop()

            self.container_id = None
            self.container_name = None

            return True

        except Exception as stop_exc:
            self.logger.error(
                f"Failed to stop container {self.container_id}: {stop_exc}"
            )
            return False

    def get_server_info(self) -> dict[str, Any]:
        """Get information about this BEDtools server."""
        return {
            "name": self.name,
            "type": self.server_type.value,
            "version": "2.30.0",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
            "capabilities": self.config.capabilities,
            "pydantic_ai_enabled": self.pydantic_ai_agent is not None,
            "session_active": self.session is not None,
        }


# Create server instance
bedtools_server = BEDToolsServer()
