"""
MACS3 MCP Server - Vendored BioinfoMCP server for ChIP-seq peak calling.

This module implements a strongly-typed MCP server for MACS3, a popular tool
for identifying transcription factor binding sites from ChIP-seq data, using
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


class MACS3Server(MCPServerBase):
    """MCP Server for MACS3 ChIP-seq peak calling tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="macs3-server",
                server_type=MCPServerType.CUSTOM,  # Will be overridden
                container_image="python:3.11-slim",
                environment_variables={"MACS3_VERSION": "3.0.0"},
                capabilities=["chip_seq", "peak_calling", "transcription_factors"],
            )
        super().__init__(config)

    @mcp_tool()
    def macs3_callpeak(
        self,
        treatment_file: str,
        control_file: str | None = None,
        output_dir: str = ".",
        name: str = "peaks",
        format: str = "BAM",
        gsize: str = "hs",
        qvalue: float = 0.05,
        pvalue: float = 0.0001,
        mfold: str = "10,30",
        nolambda: bool = False,
        broad: bool = False,
        broad_cutoff: float = 0.1,
        nomodel: bool = False,
        shift: int = 0,
        extsize: int = 200,
        keep_dup: str = "auto",
        bdg: bool = False,
        trackline: bool = False,
        spmr: bool = False,
        threads: int = 1,
    ) -> dict[str, Any]:
        """
        Call peaks from ChIP-seq data using MACS3.

        This tool identifies regions of the genome that are enriched for transcription factor binding
        or histone modifications from ChIP-seq experiments.

        Args:
            treatment_file: Treatment BAM file (ChIP sample)
            control_file: Control BAM file (input sample, optional)
            output_dir: Output directory for results
            name: Prefix for output files
            format: Format of input files (BAM, SAM, BED, etc.)
            gsize: Genome size (hs, mm, ce, dm, etc.)
            qvalue: Q-value cutoff for peak detection
            pvalue: P-value cutoff for peak detection
            mfold: Minimum and maximum fold enrichment
            nolambda: Do not use local lambda to model noise
            broad: Call broad peaks for broad marks like H3K36me3
            broad_cutoff: Cutoff for broad peak detection
            nomodel: Do not build shifting model
            shift: Base pair shift for model building
            extsize: Extension size for model building
            keep_dup: How to handle duplicate reads
            bdg: Generate bedGraph files
            trackline: Add trackline to bedGraph files
            spmr: Generate signal per million reads normalized tracks
            threads: Number of threads to use

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
        if not os.path.exists(treatment_file):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Treatment file does not exist: {treatment_file}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Treatment file not found: {treatment_file}",
            }

        if control_file and not os.path.exists(control_file):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Control file does not exist: {control_file}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Control file not found: {control_file}",
            }

        # Build command
        cmd = ["macs3", "callpeak", "-t", treatment_file]

        if control_file:
            cmd.extend(["-c", control_file])

        cmd.extend(
            [
                "-n",
                name,
                "-f",
                format,
                "-g",
                gsize,
                "-q",
                str(qvalue),
                "-p",
                str(pvalue),
                "--outdir",
                output_dir,
            ]
        )

        if mfold:
            cmd.extend(["-m", mfold])
        if nolambda:
            cmd.append("--nolambda")
        if broad:
            cmd.extend(["--broad", "--broad-cutoff", str(broad_cutoff)])
        if nomodel:
            cmd.append("--nomodel")
        if shift > 0:
            cmd.extend(["--shift", str(shift)])
        if extsize != 200:
            cmd.extend(["--extsize", str(extsize)])
        if keep_dup != "auto":
            cmd.extend(["--keep-dup", keep_dup])
        if bdg:
            cmd.append("--bdg")
        if trackline:
            cmd.append("--trackline")
        if spmr:
            cmd.append("--spmr")
        if threads > 1:
            cmd.extend(["--threads", str(threads)])

        try:
            # Execute MACS3
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=output_dir
            )

            # Get output files
            output_files = []
            try:
                output_files = [
                    f"{output_dir}/{name}_peaks.xls",
                    f"{output_dir}/{name}_peaks.narrowPeak",
                    f"{output_dir}/{name}_summits.bed",
                    f"{output_dir}/{name}_model.r",
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
                "stderr": "MACS3 not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "MACS3 not found in PATH",
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
            name="macs3_bdgcmp",
            description="Compare two bedGraph files to generate fold enrichment tracks",
            inputs={
                "treatment_bdg": "str",
                "control_bdg": "str",
                "output_dir": "str",
                "name": "str",
                "method": "str",
                "pseudocount": "float",
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
                    "description": "Compare treatment and control bedGraph files",
                    "parameters": {
                        "treatment_bdg": "/data/treatment.bdg",
                        "control_bdg": "/data/control.bdg",
                        "output_dir": "/results",
                        "name": "fold_enrichment",
                        "method": "ppois",
                    },
                }
            ],
        )
    )
    def macs3_bdgcmp(
        self,
        treatment_bdg: str,
        control_bdg: str,
        output_dir: str = ".",
        name: str = "fold_enrichment",
        method: str = "ppois",
        pseudocount: float = 1.0,
    ) -> dict[str, Any]:
        """
        Compare two bedGraph files to generate fold enrichment tracks.

        This tool compares treatment and control bedGraph files to compute
        fold enrichment and statistical significance of ChIP-seq signals.

        Args:
            treatment_bdg: Treatment bedGraph file
            control_bdg: Control bedGraph file
            output_dir: Output directory for results
            name: Prefix for output files
            method: Statistical method (ppois, qpois, FE, logFE, logLR, subtract)
            pseudocount: Pseudocount to avoid division by zero

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        if not os.path.exists(treatment_bdg):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Treatment bedGraph file does not exist: {treatment_bdg}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Treatment file not found: {treatment_bdg}",
            }

        if not os.path.exists(control_bdg):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Control bedGraph file does not exist: {control_bdg}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Control file not found: {control_bdg}",
            }

        # Build command
        cmd = [
            "macs3",
            "bdgcmp",
            "-t",
            treatment_bdg,
            "-c",
            control_bdg,
            "-o",
            f"{output_dir}/{name}",
            "-m",
            method,
        ]

        if pseudocount != 1.0:
            cmd.extend(["-p", str(pseudocount)])

        try:
            # Execute MACS3 bdgcmp
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=output_dir
            )

            # Get output files
            output_files = []
            try:
                output_files = [
                    f"{output_dir}/{name}_ppois.bdg",
                    f"{output_dir}/{name}_logLR.bdg",
                    f"{output_dir}/{name}_FE.bdg",
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
                "stderr": "MACS3 not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "MACS3 not found in PATH",
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
            name="macs3_filterdup",
            description="Filter duplicate reads from BAM files",
            inputs={
                "input_bam": "str",
                "output_bam": "str",
                "gsize": "str",
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
                    "description": "Filter duplicate reads from BAM file",
                    "parameters": {
                        "input_bam": "/data/sample.bam",
                        "output_bam": "/data/sample_filtered.bam",
                        "gsize": "hs",
                    },
                }
            ],
        )
    )
    def macs3_filterdup(
        self,
        input_bam: str,
        output_bam: str,
        gsize: str = "hs",
    ) -> dict[str, Any]:
        """
        Filter duplicate reads from BAM files.

        This tool removes duplicate reads from BAM files, which is important
        for accurate ChIP-seq peak calling.

        Args:
            input_bam: Input BAM file
            output_bam: Output BAM file with duplicates removed
            gsize: Genome size (hs, mm, ce, dm, etc.)

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
                "error": f"Input file not found: {input_bam}",
            }

        # Build command
        cmd = [
            "macs3",
            "filterdup",
            "-i",
            input_bam,
            "-o",
            output_bam,
            "-g",
            gsize,
        ]

        try:
            # Execute MACS3 filterdup
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output_bam):
                output_files = [output_bam]

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
                "stderr": "MACS3 not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "MACS3 not found in PATH",
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
        """Deploy MACS3 server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-macs3-server-{id(self)}")

            # Install MACS3
            container.with_command("bash -c 'pip install macs3 && tail -f /dev/null'")

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
        """Stop MACS3 server deployed with testcontainers."""
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
        """Get information about this MACS3 server."""
        return {
            "name": self.name,
            "type": "macs3",
            "version": "3.0.0",
            "description": "MACS3 ChIP-seq peak calling server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
