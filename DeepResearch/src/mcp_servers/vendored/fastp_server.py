"""
Fastp MCP Server - Vendored BioinfoMCP server for FASTQ preprocessing.

This module implements a strongly-typed MCP server for Fastp, an ultra-fast
all-in-one FASTQ preprocessor, using Pydantic AI patterns and testcontainers deployment.
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


class FastpServer(MCPServerBase):
    """MCP Server for Fastp FASTQ preprocessing tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="fastp-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"FASTP_VERSION": "0.23.4"},
                capabilities=[
                    "quality_control",
                    "adapter_trimming",
                    "read_filtering",
                    "preprocessing",
                ],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="fastp_process",
            description="Process FASTQ files with comprehensive quality control and adapter trimming using Fastp",
            inputs={
                "input1": "str",
                "output1": "str",
                "input2": "str | None",
                "output2": "str | None",
                "unpaired1": "str | None",
                "unpaired2": "str | None",
                "failed_out": "str | None",
                "json": "str | None",
                "html": "str | None",
                "report_title": "str",
                "threads": "int",
                "compression": "int",
                "phred64": "bool",
                "input_phred64": "bool",
                "output_phred64": "bool",
                "dont_overwrite": "bool",
                "fix_mgi_id": "bool",
                "adapter_sequence": "str | None",
                "adapter_sequence_r2": "str | None",
                "detect_adapter_for_pe": "bool",
                "trim_front1": "int",
                "trim_tail1": "int",
                "trim_front2": "int",
                "trim_tail2": "int",
                "max_len1": "int",
                "max_len2": "int",
                "trim_poly_g": "bool",
                "poly_g_min_len": "int",
                "trim_poly_x": "bool",
                "poly_x_min_len": "int",
                "cut_front": "bool",
                "cut_tail": "bool",
                "cut_window_size": "int",
                "cut_mean_quality": "int",
                "cut_front_mean_quality": "int",
                "cut_tail_mean_quality": "int",
                "cut_front_window_size": "int",
                "cut_tail_window_size": "int",
                "disable_quality_filtering": "bool",
                "qualified_quality_phred": "int",
                "unqualified_percent_limit": "int",
                "n_base_limit": "int",
                "disable_length_filtering": "bool",
                "length_required": "int",
                "length_limit": "int",
                "low_complexity_filter": "bool",
                "complexity_threshold": "float",
                "filter_by_index1": "str | None",
                "filter_by_index2": "str | None",
                "correction": "bool",
                "overlap_len_require": "int",
                "overlap_diff_limit": "int",
                "overlap_diff_percent_limit": "float",
                "umi": "bool",
                "umi_loc": "str",
                "umi_len": "int",
                "umi_prefix": "str | None",
                "umi_skip": "int",
                "overrepresentation_analysis": "bool",
                "overrepresentation_sampling": "int",
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
                    "description": "Basic FASTQ preprocessing with adapter trimming and quality filtering",
                    "parameters": {
                        "input1": "/data/sample_R1.fastq.gz",
                        "output1": "/data/sample_R1_processed.fastq.gz",
                        "input2": "/data/sample_R2.fastq.gz",
                        "output2": "/data/sample_R2_processed.fastq.gz",
                        "threads": 4,
                        "detect_adapter_for_pe": True,
                        "qualified_quality_phred": 20,
                        "length_required": 20,
                    },
                }
            ],
        )
    )
    def fastp_process(
        self,
        input1: str,
        output1: str,
        input2: str | None = None,
        output2: str | None = None,
        unpaired1: str | None = None,
        unpaired2: str | None = None,
        failed_out: str | None = None,
        json: str | None = None,
        html: str | None = None,
        report_title: str = "Fastp Report",
        threads: int = 2,
        compression: int = 4,
        phred64: bool = False,
        input_phred64: bool = False,
        output_phred64: bool = False,
        dont_overwrite: bool = False,
        fix_mgi_id: bool = False,
        adapter_sequence: str | None = None,
        adapter_sequence_r2: str | None = None,
        detect_adapter_for_pe: bool = False,
        trim_front1: int = 0,
        trim_tail1: int = 0,
        trim_front2: int = 0,
        trim_tail2: int = 0,
        max_len1: int = 0,
        max_len2: int = 0,
        trim_poly_g: bool = False,
        poly_g_min_len: int = 10,
        trim_poly_x: bool = False,
        poly_x_min_len: int = 10,
        cut_front: bool = False,
        cut_tail: bool = False,
        cut_window_size: int = 4,
        cut_mean_quality: int = 20,
        cut_front_mean_quality: int = 0,
        cut_tail_mean_quality: int = 0,
        cut_front_window_size: int = 0,
        cut_tail_window_size: int = 0,
        disable_quality_filtering: bool = False,
        qualified_quality_phred: int = 15,
        unqualified_percent_limit: int = 40,
        n_base_limit: int = 5,
        disable_length_filtering: bool = False,
        length_required: int = 15,
        length_limit: int = 0,
        low_complexity_filter: bool = False,
        complexity_threshold: float = 0.3,
        filter_by_index1: str | None = None,
        filter_by_index2: str | None = None,
        correction: bool = False,
        overlap_len_require: int = 30,
        overlap_diff_limit: int = 5,
        overlap_diff_percent_limit: float = 0.05,
        umi: bool = False,
        umi_loc: str = "none",
        umi_len: int = 0,
        umi_prefix: str | None = None,
        umi_skip: int = 0,
        overrepresentation_analysis: bool = False,
        overrepresentation_sampling: int = 20,
    ) -> dict[str, Any]:
        """
        Process FASTQ files with comprehensive quality control and adapter trimming using Fastp.

        Fastp is an ultra-fast all-in-one FASTQ preprocessor that can perform quality control,
        adapter trimming, quality filtering, per-read quality pruning, and many other operations.

        Args:
            input1: Read 1 input FASTQ file
            output1: Read 1 output FASTQ file
            input2: Read 2 input FASTQ file (for paired-end)
            output2: Read 2 output FASTQ file (for paired-end)
            unpaired1: Unpaired output for read 1
            unpaired2: Unpaired output for read 2
            failed_out: Failed reads output
            json: JSON report output
            html: HTML report output
            report_title: Title for the report
            threads: Number of threads to use
            compression: Compression level for output files
            phred64: Assume input is in Phred+64 format
            input_phred64: Assume input is in Phred+64 format
            output_phred64: Output in Phred+64 format
            dont_overwrite: Don't overwrite existing files
            fix_mgi_id: Fix MGI-specific read IDs
            adapter_sequence: Adapter sequence for read 1
            adapter_sequence_r2: Adapter sequence for read 2
            detect_adapter_for_pe: Detect adapters for paired-end reads
            trim_front1: Trim N bases from 5' end of read 1
            trim_tail1: Trim N bases from 3' end of read 1
            trim_front2: Trim N bases from 5' end of read 2
            trim_tail2: Trim N bases from 3' end of read 2
            max_len1: Maximum length for read 1
            max_len2: Maximum length for read 2
            trim_poly_g: Trim poly-G tails
            poly_g_min_len: Minimum length of poly-G to trim
            trim_poly_x: Trim poly-X tails
            poly_x_min_len: Minimum length of poly-X to trim
            cut_front: Cut front window with mean quality
            cut_tail: Cut tail window with mean quality
            cut_window_size: Window size for quality cutting
            cut_mean_quality: Mean quality threshold for cutting
            cut_front_mean_quality: Mean quality for front cutting
            cut_tail_mean_quality: Mean quality for tail cutting
            cut_front_window_size: Window size for front cutting
            cut_tail_window_size: Window size for tail cutting
            disable_quality_filtering: Disable quality filtering
            qualified_quality_phred: Minimum Phred quality for qualified bases
            unqualified_percent_limit: Maximum percentage of unqualified bases
            n_base_limit: Maximum number of N bases allowed
            disable_length_filtering: Disable length filtering
            length_required: Minimum read length required
            length_limit: Maximum read length allowed
            low_complexity_filter: Enable low complexity filter
            complexity_threshold: Complexity threshold
            filter_by_index1: Filter by index for read 1
            filter_by_index2: Filter by index for read 2
            correction: Enable error correction for paired-end reads
            overlap_len_require: Minimum overlap length for correction
            overlap_diff_limit: Maximum difference for correction
            overlap_diff_percent_limit: Maximum difference percentage for correction
            umi: Enable UMI processing
            umi_loc: UMI location (none, index1, index2, read1, read2, per_index, per_read)
            umi_len: UMI length
            umi_prefix: UMI prefix
            umi_skip: Number of bases to skip for UMI
            overrepresentation_analysis: Enable overrepresentation analysis
            overrepresentation_sampling: Sampling rate for overrepresentation analysis

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        if not os.path.exists(input1):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Input file 1 does not exist: {input1}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Input file 1 not found: {input1}",
            }

        if input2 and not os.path.exists(input2):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Input file 2 does not exist: {input2}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Input file 2 not found: {input2}",
            }

        # Build command
        cmd = ["fastp"]

        # Input/output files
        cmd.extend(["-i", input1, "-o", output1])
        if input2 and output2:
            cmd.extend(["-I", input2, "-O", output2])
        if unpaired1:
            cmd.extend(["--unpaired1", unpaired1])
        if unpaired2:
            cmd.extend(["--unpaired2", unpaired2])
        if failed_out:
            cmd.extend(["--failed_out", failed_out])

        # Report files
        if json:
            cmd.extend(["-j", json])
        if html:
            cmd.extend(["-h", html])
        cmd.extend(["-R", report_title])

        # Basic options
        cmd.extend(["-w", str(threads)])
        cmd.extend(["-z", str(compression)])

        if phred64 or input_phred64:
            cmd.append("--phred64")
        if output_phred64:
            cmd.append("--output_phred64")
        if dont_overwrite:
            cmd.append("--dont_overwrite")
        if fix_mgi_id:
            cmd.append("--fix_mgi_id")

        # Adapter trimming
        if adapter_sequence:
            cmd.extend(["-a", adapter_sequence])
        if adapter_sequence_r2:
            cmd.extend(["-A", adapter_sequence_r2])
        if detect_adapter_for_pe:
            cmd.append("--detect_adapter_for_pe")

        # Trimming options
        if trim_front1 > 0:
            cmd.extend(["-f", str(trim_front1)])
        if trim_tail1 > 0:
            cmd.extend(["-t", str(trim_tail1)])
        if trim_front2 > 0:
            cmd.extend(["-F", str(trim_front2)])
        if trim_tail2 > 0:
            cmd.extend(["-T", str(trim_tail2)])
        if max_len1 > 0:
            cmd.extend(["--max_len1", str(max_len1)])
        if max_len2 > 0:
            cmd.extend(["--max_len2", str(max_len2)])
        if trim_poly_g:
            cmd.extend(["--trim_poly_g", "--poly_g_min_len", str(poly_g_min_len)])
        if trim_poly_x:
            cmd.extend(["--trim_poly_x", "--poly_x_min_len", str(poly_x_min_len)])

        # Quality cutting
        if cut_front:
            cmd.extend(
                [
                    "--cut_front",
                    "--cut_front_window_size",
                    str(cut_front_window_size or cut_window_size),
                    "--cut_front_mean_quality",
                    str(cut_front_mean_quality or cut_mean_quality),
                ]
            )
        if cut_tail:
            cmd.extend(
                [
                    "--cut_tail",
                    "--cut_tail_window_size",
                    str(cut_tail_window_size or cut_window_size),
                    "--cut_tail_mean_quality",
                    str(cut_tail_mean_quality or cut_mean_quality),
                ]
            )

        # Quality filtering
        if not disable_quality_filtering:
            cmd.extend(
                [
                    "--qualified_quality_phred",
                    str(qualified_quality_phred),
                    "--unqualified_percent_limit",
                    str(unqualified_percent_limit),
                    "--n_base_limit",
                    str(n_base_limit),
                ]
            )

        # Length filtering
        if not disable_length_filtering:
            cmd.extend(["--length_required", str(length_required)])
            if length_limit > 0:
                cmd.extend(["--length_limit", str(length_limit)])

        # Low complexity filtering
        if low_complexity_filter:
            cmd.extend(
                [
                    "--low_complexity_filter",
                    "--complexity_threshold",
                    str(complexity_threshold),
                ]
            )

        # Index filtering
        if filter_by_index1:
            cmd.extend(["--filter_by_index1", filter_by_index1])
        if filter_by_index2:
            cmd.extend(["--filter_by_index2", filter_by_index2])

        # Error correction
        if correction:
            cmd.extend(
                [
                    "--correction",
                    "--overlap_len_require",
                    str(overlap_len_require),
                    "--overlap_diff_limit",
                    str(overlap_diff_limit),
                    "--overlap_diff_percent_limit",
                    str(overlap_diff_percent_limit),
                ]
            )

        # UMI processing
        if umi:
            cmd.extend(["--umi", "--umi_loc", umi_loc])
            if umi_len > 0:
                cmd.extend(["--umi_len", str(umi_len)])
            if umi_prefix:
                cmd.extend(["--umi_prefix", umi_prefix])
            if umi_skip > 0:
                cmd.extend(["--umi_skip", str(umi_skip)])

        # Overrepresentation analysis
        if overrepresentation_analysis:
            cmd.extend(
                [
                    "--overrepresentation_analysis",
                    "--overrepresentation_sampling",
                    str(overrepresentation_sampling),
                ]
            )

        try:
            # Execute Fastp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output1):
                output_files.append(output1)
            if output2 and os.path.exists(output2):
                output_files.append(output2)
            if unpaired1 and os.path.exists(unpaired1):
                output_files.append(unpaired1)
            if unpaired2 and os.path.exists(unpaired2):
                output_files.append(unpaired2)
            if failed_out and os.path.exists(failed_out):
                output_files.append(failed_out)
            if json and os.path.exists(json):
                output_files.append(json)
            if html and os.path.exists(html):
                output_files.append(html)

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
                "stderr": "Fastp not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Fastp not found in PATH",
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
        """Deploy Fastp server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-fastp-server-{id(self)}")

            # Install Fastp
            container.with_command(
                "bash -c 'apt-get update && apt-get install -y fastp && tail -f /dev/null'"
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
        """Stop Fastp server deployed with testcontainers."""
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
        """Get information about this Fastp server."""
        return {
            "name": self.name,
            "type": "fastp",
            "version": "0.23.4",
            "description": "Fastp FASTQ preprocessing server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
