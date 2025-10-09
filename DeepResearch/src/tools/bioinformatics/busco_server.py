"""
BUSCO MCP Server - Vendored BioinfoMCP server for genome completeness assessment.

This module implements a strongly-typed MCP server for BUSCO (Benchmarking
Universal Single-Copy Orthologs), a tool for assessing genome assembly and
annotation completeness, using Pydantic AI patterns and testcontainers deployment.
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


class BUSCOServer(MCPServerBase):
    """MCP Server for BUSCO genome completeness assessment tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="busco-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"BUSCO_VERSION": "5.4.7"},
                capabilities=[
                    "genome_assessment",
                    "completeness_analysis",
                    "annotation_quality",
                ],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="busco_run",
            description="Run BUSCO completeness assessment on genome assembly or annotation",
            inputs={
                "input_file": "str",
                "output_dir": "str",
                "mode": "str",
                "lineage_dataset": "str",
                "cpu": "int",
                "force": "bool",
                "restart": "bool",
                "download_path": "str | None",
                "datasets_version": "str | None",
                "offline": "bool",
                "augustus": "bool",
                "augustus_species": "str | None",
                "augustus_parameters": "str | None",
                "meta": "bool",
                "metaeuk": "bool",
                "metaeuk_parameters": "str | None",
                "miniprot": "bool",
                "miniprot_parameters": "str | None",
                "long": "bool",
                "evalue": "float",
                "limit": "int",
                "config": "str | None",
                "tarzip": "bool",
                "quiet": "bool",
                "out": "str | None",
                "out_path": "str | None",
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
                    "description": "Assess genome assembly completeness using BUSCO",
                    "parameters": {
                        "input_file": "/data/genome.fa",
                        "output_dir": "/results/busco",
                        "mode": "genome",
                        "lineage_dataset": "bacteria_odb10",
                        "cpu": 4,
                    },
                }
            ],
        )
    )
    def busco_run(
        self,
        input_file: str,
        output_dir: str,
        mode: str,
        lineage_dataset: str,
        cpu: int = 1,
        force: bool = False,
        restart: bool = False,
        download_path: str | None = None,
        datasets_version: str | None = None,
        offline: bool = False,
        augustus: bool = False,
        augustus_species: str | None = None,
        augustus_parameters: str | None = None,
        meta: bool = False,
        metaeuk: bool = False,
        metaeuk_parameters: str | None = None,
        miniprot: bool = False,
        miniprot_parameters: str | None = None,
        long: bool = False,
        evalue: float = 0.001,
        limit: int = 3,
        config: str | None = None,
        tarzip: bool = False,
        quiet: bool = False,
        out: str | None = None,
        out_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Run BUSCO completeness assessment on genome assembly or annotation.

        BUSCO assesses genome assembly and annotation completeness by searching for
        Benchmarking Universal Single-Copy Orthologs.

        Args:
            input_file: Input sequence file (FASTA format)
            output_dir: Output directory for results
            mode: Analysis mode (genome, proteins, transcriptome)
            lineage_dataset: Lineage dataset to use
            cpu: Number of CPUs to use
            force: Force rerun even if output directory exists
            restart: Restart from checkpoint
            download_path: Path to download lineage datasets
            datasets_version: Version of datasets to use
            offline: Run in offline mode
            augustus: Use Augustus gene prediction
            augustus_species: Augustus species model
            augustus_parameters: Additional Augustus parameters
            meta: Run in metagenome mode
            metaeuk: Use MetaEuk for protein prediction
            metaeuk_parameters: MetaEuk parameters
            miniprot: Use Miniprot for protein prediction
            miniprot_parameters: Miniprot parameters
            long: Enable long mode for large genomes
            evalue: E-value threshold for BLAST searches
            limit: Maximum number of candidate genes per BUSCO
            config: Configuration file
            tarzip: Compress output directory
            quiet: Suppress verbose output
            out: Output prefix
            out_path: Output path (alternative to output_dir)

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
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

        # Validate mode
        valid_modes = ["genome", "proteins", "transcriptome"]
        if mode not in valid_modes:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Invalid mode: {mode}",
            }

        # Build command
        cmd = [
            "busco",
            "--in",
            input_file,
            "--out_path",
            output_dir,
            "--mode",
            mode,
            "--lineage_dataset",
            lineage_dataset,
            "--cpu",
            str(cpu),
        ]

        if force:
            cmd.append("--force")
        if restart:
            cmd.append("--restart")
        if download_path:
            cmd.extend(["--download_path", download_path])
        if datasets_version:
            cmd.extend(["--datasets_version", datasets_version])
        if offline:
            cmd.append("--offline")
        if augustus:
            cmd.append("--augustus")
        if augustus_species:
            cmd.extend(["--augustus_species", augustus_species])
        if augustus_parameters:
            cmd.extend(["--augustus_parameters", augustus_parameters])
        if meta:
            cmd.append("--meta")
        if metaeuk:
            cmd.append("--metaeuk")
        if metaeuk_parameters:
            cmd.extend(["--metaeuk_parameters", metaeuk_parameters])
        if miniprot:
            cmd.append("--miniprot")
        if miniprot_parameters:
            cmd.extend(["--miniprot_parameters", miniprot_parameters])
        if long:
            cmd.append("--long")
        if evalue != 0.001:
            cmd.extend(["--evalue", str(evalue)])
        if limit != 3:
            cmd.extend(["--limit", str(limit)])
        if config:
            cmd.extend(["--config", config])
        if tarzip:
            cmd.append("--tarzip")
        if quiet:
            cmd.append("--quiet")
        if out:
            cmd.extend(["--out", out])
        if out_path:
            cmd.extend(["--out_path", out_path])

        try:
            # Execute BUSCO
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=output_dir
            )

            # Get output files
            output_files = []
            try:
                # BUSCO creates several output files
                busco_output_dir = os.path.join(output_dir, "busco_downloads")
                if os.path.exists(busco_output_dir):
                    output_files.append(busco_output_dir)

                # Look for short_summary files
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.startswith("short_summary"):
                            output_files.append(os.path.join(root, file))

                # Look for other important output files
                important_files = [
                    "full_table.tsv",
                    "missing_busco_list.tsv",
                    "run_busco.log",
                ]
                for file in important_files:
                    file_path = os.path.join(output_dir, file)
                    if os.path.exists(file_path):
                        output_files.append(file_path)

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
                "stderr": "BUSCO not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "BUSCO not found in PATH",
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
            name="busco_download",
            description="Download BUSCO lineage datasets",
            inputs={
                "lineage_dataset": "str",
                "download_path": "str | None",
                "datasets_version": "str | None",
                "force": "bool",
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
                    "description": "Download bacterial BUSCO dataset",
                    "parameters": {
                        "lineage_dataset": "bacteria_odb10",
                        "download_path": "/data/busco_datasets",
                    },
                }
            ],
        )
    )
    def busco_download(
        self,
        lineage_dataset: str,
        download_path: str | None = None,
        datasets_version: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Download BUSCO lineage datasets.

        This tool downloads specific BUSCO lineage datasets for later use.

        Args:
            lineage_dataset: Lineage dataset to download
            download_path: Path to download datasets
            datasets_version: Version of datasets to download
            force: Force download even if dataset exists

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Build command
        cmd = ["busco", "--download", lineage_dataset]

        if download_path:
            cmd.extend(["--download_path", download_path])
        if datasets_version:
            cmd.extend(["--datasets_version", datasets_version])
        if force:
            cmd.append("--force")

        try:
            # Execute BUSCO download
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if download_path and os.path.exists(download_path):
                output_files.append(download_path)

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
                "stderr": "BUSCO not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "BUSCO not found in PATH",
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
        """Deploy BUSCO server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-busco-server-{id(self)}")

            # Install BUSCO
            container.with_command("bash -c 'pip install busco && tail -f /dev/null'")

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
        """Stop BUSCO server deployed with testcontainers."""
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
        """Get information about this BUSCO server."""
        return {
            "name": self.name,
            "type": "busco",
            "version": "5.4.7",
            "description": "BUSCO genome completeness assessment server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
