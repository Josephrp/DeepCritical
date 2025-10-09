"""
Picard MCP Server - Vendored BioinfoMCP server for SAM/BAM processing.

This module implements a strongly-typed MCP server for Picard, a set of command line
tools for manipulating high-throughput sequencing data and formats, using Pydantic AI
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


class PicardServer(MCPServerBase):
    """MCP Server for Picard SAM/BAM processing tools with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="picard-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"PICARD_VERSION": "3.0.0"},
                capabilities=["sam_processing", "bam_processing", "quality_control"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="picard_mark_duplicates",
            description="Mark duplicate reads in BAM files using Picard",
            inputs={
                "input_bam": "str",
                "output_bam": "str",
                "metrics_file": "str",
                "remove_duplicates": "bool",
                "assume_sorted": "bool",
                "duplicate_scoring_strategy": "str",
                "optical_duplicate_pixel_distance": "int",
                "validation_stringency": "str",
                "tmp_dir": "str | None",
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
                    "description": "Mark duplicates in a BAM file",
                    "parameters": {
                        "input_bam": "/data/sample.bam",
                        "output_bam": "/data/sample_marked.bam",
                        "metrics_file": "/data/mark_duplicates_metrics.txt",
                        "remove_duplicates": False,
                    },
                }
            ],
        )
    )
    def picard_mark_duplicates(
        self,
        input_bam: str,
        output_bam: str,
        metrics_file: str,
        remove_duplicates: bool = False,
        assume_sorted: bool = True,
        duplicate_scoring_strategy: str = "SUM_OF_BASE_QUALITIES",
        optical_duplicate_pixel_distance: int = 100,
        validation_stringency: str = "SILENT",
        tmp_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Mark duplicate reads in BAM files using Picard.

        This tool identifies and marks duplicate reads in BAM files, which is crucial
        for accurate downstream analysis in next-generation sequencing pipelines.

        Args:
            input_bam: Input BAM file
            output_bam: Output BAM file with duplicates marked
            metrics_file: File to write duplicate metrics
            remove_duplicates: Remove duplicate reads instead of marking
            assume_sorted: Assume input BAM is coordinate sorted
            duplicate_scoring_strategy: Strategy for scoring duplicates
            optical_duplicate_pixel_distance: Maximum offset for optical duplicates
            validation_stringency: Validation stringency level
            tmp_dir: Temporary directory for intermediate files

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
        cmd = ["picard", "MarkDuplicates"]

        cmd.extend(["INPUT=" + input_bam])
        cmd.extend(["OUTPUT=" + output_bam])
        cmd.extend(["METRICS_FILE=" + metrics_file])

        if remove_duplicates:
            cmd.append("REMOVE_DUPLICATES=true")
        else:
            cmd.append("REMOVE_DUPLICATES=false")

        if assume_sorted:
            cmd.append("ASSUME_SORTED=true")
        else:
            cmd.append("ASSUME_SORTED=false")

        cmd.extend(["DUPLICATE_SCORING_STRATEGY=" + duplicate_scoring_strategy])
        cmd.extend(
            [
                "OPTICAL_DUPLICATE_PIXEL_DISTANCE="
                + str(optical_duplicate_pixel_distance)
            ]
        )
        cmd.extend(["VALIDATION_STRINGENCY=" + validation_stringency])

        if tmp_dir:
            cmd.extend(["TMP_DIR=" + tmp_dir])

        try:
            # Execute Picard MarkDuplicates
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output_bam):
                output_files.append(output_bam)
            if os.path.exists(metrics_file):
                output_files.append(metrics_file)

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
                "stderr": "Picard not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Picard not found in PATH",
            }
        except Exception as e:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Unexpected error: {e}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Unexpected error: {e}",
            }

    @mcp_tool(
        MCPToolSpec(
            name="picard_collect_alignment_summary_metrics",
            description="Collect alignment summary metrics using Picard",
            inputs={
                "input_bam": "str",
                "output_file": "str",
                "reference_sequence": "str | None",
                "metric_accumulation_level": "str",
                "assume_sorted": "bool",
                "adapter_sequence": "str | None",
                "max_insert_size": "int",
                "is_bisulfite_sequenced": "bool",
                "validation_stringency": "str",
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
                    "description": "Collect alignment summary metrics",
                    "parameters": {
                        "input_bam": "/data/sample.bam",
                        "output_file": "/data/alignment_metrics.txt",
                        "metric_accumulation_level": "SAMPLE",
                        "reference_sequence": "/data/hg38.fa",
                    },
                }
            ],
        )
    )
    def picard_collect_alignment_summary_metrics(
        self,
        input_bam: str,
        output_file: str,
        reference_sequence: str | None = None,
        metric_accumulation_level: str = "SAMPLE",
        assume_sorted: bool = True,
        adapter_sequence: str | None = None,
        max_insert_size: int = 100000,
        is_bisulfite_sequenced: bool = False,
        validation_stringency: str = "SILENT",
    ) -> dict[str, Any]:
        """
        Collect alignment summary metrics using Picard.

        This tool collects summary metrics about the alignment of reads to a reference genome,
        providing important quality control information.

        Args:
            input_bam: Input BAM file
            output_file: Output metrics file
            reference_sequence: Reference FASTA file
            metric_accumulation_level: Level at which to accumulate metrics
            assume_sorted: Assume input BAM is coordinate sorted
            adapter_sequence: Adapter sequence to trim
            max_insert_size: Maximum insert size for paired reads
            is_bisulfite_sequenced: Whether the data is bisulfite sequenced
            validation_stringency: Validation stringency level

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
        cmd = ["picard", "CollectAlignmentSummaryMetrics"]

        cmd.extend(["INPUT=" + input_bam])
        cmd.extend(["OUTPUT=" + output_file])

        if reference_sequence:
            cmd.extend(["REFERENCE_SEQUENCE=" + reference_sequence])

        cmd.extend(["METRIC_ACCUMULATION_LEVEL=" + metric_accumulation_level])

        if assume_sorted:
            cmd.append("ASSUME_SORTED=true")
        else:
            cmd.append("ASSUME_SORTED=false")

        if adapter_sequence:
            cmd.extend(["ADAPTER_SEQUENCE=" + adapter_sequence])

        cmd.extend(["MAX_INSERT_SIZE=" + str(max_insert_size)])

        if is_bisulfite_sequenced:
            cmd.append("IS_BISULFITE_SEQUENCED=true")
        else:
            cmd.append("IS_BISULFITE_SEQUENCED=false")

        cmd.extend(["VALIDATION_STRINGENCY=" + validation_stringency])

        try:
            # Execute Picard CollectAlignmentSummaryMetrics
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output_file):
                output_files.append(output_file)

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
                "stderr": "Picard not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Picard not found in PATH",
            }
        except Exception as e:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Unexpected error: {e}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Unexpected error: {e}",
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy Picard server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-picard-server-{id(self)}")

            # Install Picard
            container.with_command(
                "bash -c 'apt-get update && apt-get install -y openjdk-11-jdk wget && wget -q https://github.com/broadinstitute/picard/releases/download/3.0.0/picard.jar -O /usr/local/bin/picard.jar && echo \"java -jar /usr/local/bin/picard.jar $@\" > /usr/local/bin/picard && chmod +x /usr/local/bin/picard && tail -f /dev/null'"
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
        """Stop Picard server deployed with testcontainers."""
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
        """Get information about this Picard server."""
        return {
            "name": self.name,
            "type": "picard",
            "version": "3.0.0",
            "description": "Picard SAM/BAM processing server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
