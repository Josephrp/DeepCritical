"""
HOMER MCP Server - Vendored BioinfoMCP server for motif analysis.

This module implements a strongly-typed MCP server for HOMER, a suite of tools
for motif discovery and next-generation sequencing analysis, using Pydantic AI
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


class HOMERServer(MCPServerBase):
    """MCP Server for HOMER motif analysis tools with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="homer-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"HOMER_VERSION": "4.11"},
                capabilities=["motif_discovery", "chip_seq", "ngs_analysis"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="homer_findMotifs",
            description="Find motifs in genomic regions using HOMER",
            inputs={
                "input_file": "str",
                "output_dir": "str",
                "genome": "str",
                "size": "str",
                "mask": "bool",
                "bg": "str | None",
                "len": "str",
                "S": "int",
                "mis": "int",
                "norevopp": "bool",
                "nomotif": "bool",
                "bits": "bool",
                "nocheck": "bool",
                "p": "int",
            },
            outputs={
                "command_executed": "str",
                "stdout": "str",
                "stderr": "str",
                "output_files": "list[str]",
                "exit_code": "int",
            },
            server_type=MCPServerType.CUSTOM,
            command_template="findMotifs.pl {input_file} {genome} {output_dir} -size {size} {mask_flag} {bg_flag} -len {len} -S {S} -mis {mis} {norevopp_flag} {nomotif_flag} {bits_flag} {nocheck_flag} -p {p}",
            examples=[
                {
                    "description": "Basic motif discovery in ChIP-seq peaks",
                    "parameters": {
                        "input_file": "/data/peaks.bed",
                        "output_dir": "/results",
                        "genome": "hg38",
                        "size": "200",
                        "len": "8,10,12",
                    },
                }
            ],
        )
    )
    def homer_findMotifs(
        self,
        input_file: str,
        output_dir: str,
        genome: str,
        size: str = "200",
        mask: bool = False,
        bg: str | None = None,
        len: str = "8,10,12",
        S: int = 1,
        mis: int = 2,
        norevopp: bool = False,
        nomotif: bool = False,
        bits: bool = False,
        nocheck: bool = False,
        p: int = 1,
    ) -> dict[str, Any]:
        """
        Find motifs in genomic regions using HOMER.

        This tool discovers enriched motifs in a set of genomic regions (peaks, promoters, etc.)
        compared to background regions using HOMER's motif discovery algorithms.

        Args:
            input_file: Input BED file with genomic regions
            output_dir: Output directory for results
            genome: Genome assembly (hg38, mm10, etc.)
            size: Size of regions to analyze
            mask: Mask repeats and low complexity regions
            bg: Background file (optional)
            len: Motif lengths to search (comma-separated)
            S: Number of motifs to find
            mis: Maximum mismatches allowed
            norevopp: Don't search reverse complement strand
            nomotif: Skip motif finding step
            bits: Output results in bits format
            nocheck: Skip sequence validation
            p: Number of parallel processes

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

        if bg and not os.path.exists(bg):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Background file does not exist: {bg}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Background file not found: {bg}",
            }

        # Build command
        cmd = ["findMotifs.pl", input_file, genome, output_dir]

        cmd.extend(["-size", size])
        if mask:
            cmd.append("-mask")
        if bg:
            cmd.extend(["-bg", bg])
        cmd.extend(["-len", len])
        cmd.extend(["-S", str(S)])
        cmd.extend(["-mis", str(mis)])
        if norevopp:
            cmd.append("-norevopp")
        if nomotif:
            cmd.append("-nomotif")
        if bits:
            cmd.append("-bits")
        if nocheck:
            cmd.append("-nocheck")
        cmd.extend(["-p", str(p)])

        try:
            # Execute HOMER findMotifs
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, cwd=output_dir
            )

            # Get output files
            output_files = []
            try:
                # HOMER typically creates several output files
                base_name = os.path.basename(input_file)
                base_name = base_name.replace(".bed", "").replace(".txt", "")
                output_files = [
                    f"{output_dir}/homerResults.html",
                    f"{output_dir}/knownResults.html",
                    f"{output_dir}/motifFindingParameters.txt",
                    f"{output_dir}/{base_name}_motifs.txt",
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
                "stderr": "HOMER not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "HOMER not found in PATH",
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
            name="homer_annotatePeaks",
            description="Annotate peaks with genomic features using HOMER",
            inputs={
                "input_file": "str",
                "genome": "str",
                "output_file": "str",
                "gid": "bool",
                "gname": "bool",
                "tss": "bool",
                "gsize": "int",
                "bed": "bool",
                "gtf": "str | None",
                "go": "bool",
                "genomeOntology": "bool",
                "len": "int",
                "log": "bool",
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
                    "description": "Annotate ChIP-seq peaks with genomic features",
                    "parameters": {
                        "input_file": "/data/peaks.bed",
                        "genome": "hg38",
                        "output_file": "/data/annotated_peaks.txt",
                        "gname": True,
                        "tss": True,
                    },
                }
            ],
        )
    )
    def homer_annotatePeaks(
        self,
        input_file: str,
        genome: str,
        output_file: str,
        gid: bool = False,
        gname: bool = False,
        tss: bool = False,
        gsize: int = 1000,
        bed: bool = False,
        gtf: str | None = None,
        go: bool = False,
        genomeOntology: bool = False,
        len: int = 1000,
        log: bool = False,
    ) -> dict[str, Any]:
        """
        Annotate peaks with genomic features using HOMER.

        This tool annotates genomic regions with nearby genes, TSS, and other
        genomic features to understand the biological context of peaks.

        Args:
            input_file: Input BED file with peaks
            genome: Genome assembly
            output_file: Output annotated file
            gid: Include gene IDs
            gname: Include gene names
            tss: Include distance to TSS
            gsize: Gene size for annotation
            bed: Output in BED format
            gtf: Custom GTF file
            go: Include GO annotations
            genomeOntology: Include genome ontology
            len: Length of annotation region
            log: Use log scale for distances

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

        # Build command
        cmd = ["annotatePeaks.pl", input_file, genome, ">", output_file]

        if gid:
            cmd.append("-gid")
        if gname:
            cmd.append("-gname")
        if tss:
            cmd.append("-tss")
        if gsize != 1000:
            cmd.extend(["-gsize", str(gsize)])
        if bed:
            cmd.append("-bed")
        if gtf:
            cmd.extend(["-gtf", gtf])
        if go:
            cmd.append("-go")
        if genomeOntology:
            cmd.append("-genomeOntology")
        if len != 1000:
            cmd.extend(["-len", str(len)])
        if log:
            cmd.append("-log")

        try:
            # Execute HOMER annotatePeaks
            result = subprocess.run(
                " ".join(cmd),
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(output_file):
                output_files = [output_file]

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
                "stderr": "HOMER not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "HOMER not found in PATH",
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
            name="homer_mergePeaks",
            description="Merge overlapping peaks using HOMER",
            inputs={
                "input_files": "list[str]",
                "output_file": "str",
                "d": "int",
                "strand": "bool",
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
                    "description": "Merge overlapping ChIP-seq peaks",
                    "parameters": {
                        "input_files": ["/data/peaks1.bed", "/data/peaks2.bed"],
                        "output_file": "/data/merged_peaks.bed",
                        "d": 100,
                    },
                }
            ],
        )
    )
    def homer_mergePeaks(
        self,
        input_files: list[str],
        output_file: str,
        d: int = 100,
        strand: bool = False,
    ) -> dict[str, Any]:
        """
        Merge overlapping peaks using HOMER.

        This tool merges overlapping peaks from multiple samples or replicates
        to create consensus peak sets.

        Args:
            input_files: List of input BED files with peaks
            output_file: Output merged peaks file
            d: Maximum distance to merge peaks
            strand: Consider strand information for merging

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
        cmd = ["mergePeaks", "-d", str(d)]

        if strand:
            cmd.append("-strand")

        cmd.extend(input_files)

        try:
            # Execute HOMER mergePeaks with output redirection
            with open(output_file, "w") as outfile:
                result = subprocess.run(
                    cmd,
                    stdout=outfile,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

            # Get output files
            output_files = []
            if os.path.exists(output_file):
                output_files = [output_file]

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
                "stderr": "HOMER not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "HOMER not found in PATH",
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
        """Deploy HOMER server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-homer-server-{id(self)}")

            # Install HOMER
            container.with_command(
                "bash -c 'pip install homer-py && tail -f /dev/null'"
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
        """Stop HOMER server deployed with testcontainers."""
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
        """Get information about this HOMER server."""
        return {
            "name": self.name,
            "type": "homer",
            "version": "4.11",
            "description": "HOMER motif analysis server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
