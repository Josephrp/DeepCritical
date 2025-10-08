"""
MultiQC MCP Server - Vendored BioinfoMCP server for report generation.

This module implements a strongly-typed MCP server for MultiQC, a tool for
aggregating results from bioinformatics tools into a single report, using
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


class MultiQCServer(MCPServerBase):
    """MCP Server for MultiQC report generation tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="multiqc-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"MULTIQC_VERSION": "1.14"},
                capabilities=["report_generation", "quality_control", "visualization"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="multiqc_run",
            description="Generate MultiQC report from bioinformatics tool outputs",
            inputs={
                "input_dir": "str",
                "output_dir": "str",
                "filename": "str",
                "title": "str",
                "comment": "str",
                "force": "bool",
                "ignore_samples": "str",
                "sample_names": "str",
                "replace_names": "str",
                "exclude": "str",
                "include": "str",
                "zip_data": "bool",
                "export_plots": "bool",
                "flat": "bool",
                "interactive": "bool",
                "pdf": "bool",
                "no_report": "bool",
                "template": "str",
                "config": "str",
                "cl_config": "str",
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
                    "description": "Generate MultiQC report from analysis results",
                    "parameters": {
                        "input_dir": "/data/analysis_results",
                        "output_dir": "/data/reports",
                        "filename": "multiqc_report",
                        "title": "NGS Analysis Report",
                    },
                }
            ],
        )
    )
    def multiqc_run(
        self,
        input_dir: str,
        output_dir: str,
        filename: str = "multiqc_report",
        title: str = "",
        comment: str = "",
        force: bool = False,
        ignore_samples: str = "",
        sample_names: str = "",
        replace_names: str = "",
        exclude: str = "",
        include: str = "",
        zip_data: bool = False,
        export_plots: bool = False,
        flat: bool = False,
        interactive: bool = False,
        pdf: bool = False,
        no_report: bool = False,
        template: str = "",
        config: str = "",
        cl_config: str = "",
    ) -> dict[str, Any]:
        """
        Generate MultiQC report from bioinformatics tool outputs.

        This tool aggregates results from multiple bioinformatics tools into
        a single, comprehensive HTML report with interactive plots and tables.

        Args:
            input_dir: Directory containing bioinformatics tool outputs
            output_dir: Directory to save the report
            filename: Base name for output files
            title: Report title
            comment: Report comment/description
            force: Overwrite existing reports
            ignore_samples: Samples to ignore (comma-separated)
            sample_names: Rename samples (comma-separated old:new pairs)
            replace_names: Replace sample names (comma-separated old:new pairs)
            exclude: Modules to exclude (comma-separated)
            include: Modules to include (comma-separated)
            zip_data: Compress data directory
            export_plots: Export plots as static images
            flat: Flatten directory structure
            interactive: Generate interactive plots
            pdf: Generate PDF report
            no_report: Skip HTML report generation
            template: Custom template
            config: Configuration file
            cl_config: Command-line configuration

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input directory exists
        if not os.path.exists(input_dir):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Input directory does not exist: {input_dir}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Input directory not found: {input_dir}",
            }

        # Build command
        cmd = ["multiqc", input_dir, "--outdir", output_dir, "--filename", filename]

        if title:
            cmd.extend(["--title", title])
        if comment:
            cmd.extend(["--comment", comment])
        if force:
            cmd.append("--force")
        if ignore_samples:
            cmd.extend(["--ignore-samples", ignore_samples])
        if sample_names:
            cmd.extend(["--sample-names", sample_names])
        if replace_names:
            cmd.extend(["--replace-names", replace_names])
        if exclude:
            cmd.extend(["--exclude", exclude])
        if include:
            cmd.extend(["--include", include])
        if zip_data:
            cmd.append("--zip-data")
        if export_plots:
            cmd.append("--export-plots")
        if flat:
            cmd.append("--flat")
        if interactive:
            cmd.append("--interactive")
        if pdf:
            cmd.append("--pdf")
        if no_report:
            cmd.append("--no-report")
        if template:
            cmd.extend(["--template", template])
        if config:
            cmd.extend(["--config", config])
        if cl_config:
            cmd.extend(["--cl-config", cl_config])

        try:
            # Execute MultiQC report generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # MultiQC typically creates HTML report and data directory
                html_report = f"{output_dir}/{filename}.html"
                data_dir = f"{output_dir}/{filename}_data"

                if os.path.exists(html_report):
                    output_files.append(html_report)
                if os.path.exists(data_dir):
                    output_files.append(data_dir)
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
                "stderr": "MultiQC not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "MultiQC not found in PATH",
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
            name="multiqc_modules",
            description="List available MultiQC modules",
            inputs={
                "search_pattern": "str",
            },
            outputs={
                "command_executed": "str",
                "stdout": "str",
                "stderr": "str",
                "modules": "list[str]",
                "exit_code": "int",
            },
            server_type=MCPServerType.CUSTOM,
            examples=[
                {
                    "description": "List all available MultiQC modules",
                    "parameters": {
                        "search_pattern": "*",
                    },
                }
            ],
        )
    )
    def multiqc_modules(
        self,
        search_pattern: str = "*",
    ) -> dict[str, Any]:
        """
        List available MultiQC modules.

        This tool lists all available MultiQC modules that can be used
        to generate reports from different bioinformatics tools.

        Args:
            search_pattern: Pattern to search for modules

        Returns:
            Dictionary containing command executed, stdout, stderr, modules list, and exit code
        """
        # Build command
        cmd = ["multiqc", "--list-modules"]

        if search_pattern != "*":
            cmd.extend(["--search", search_pattern])

        try:
            # Execute MultiQC modules list
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Parse modules from output
            modules = []
            try:
                lines = result.stdout.split("\n")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("Available modules:"):
                        modules.append(line)
            except Exception:
                pass

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "modules": modules,
                "exit_code": result.returncode,
                "success": result.returncode == 0,
            }

        except FileNotFoundError:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": "MultiQC not found in PATH",
                "modules": [],
                "exit_code": -1,
                "success": False,
                "error": "MultiQC not found in PATH",
            }
        except Exception as e:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": str(e),
                "modules": [],
                "exit_code": -1,
                "success": False,
                "error": str(e),
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy MultiQC server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-multiqc-server-{id(self)}")

            # Install MultiQC
            container.with_command("bash -c 'pip install multiqc && tail -f /dev/null'")

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
        """Stop MultiQC server deployed with testcontainers."""
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
        """Get information about this MultiQC server."""
        return {
            "name": self.name,
            "type": "multiqc",
            "version": "1.14",
            "description": "MultiQC report generation server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
