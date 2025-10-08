"""
HISAT2 MCP Server - Vendored BioinfoMCP server for RNA-seq alignment.

This module implements a strongly-typed MCP server for HISAT2, a fast and
sensitive alignment program for mapping next-generation sequencing reads
against genomes, using Pydantic AI patterns and testcontainers deployment.
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


class HISAT2Server(MCPServerBase):
    """MCP Server for HISAT2 RNA-seq alignment tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="hisat2-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"HISAT2_VERSION": "2.2.1"},
                capabilities=["rna_seq", "alignment", "spliced_alignment"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="hisat2_build",
            description="Build HISAT2 index from genome FASTA file",
            inputs={
                "reference": "str",
                "index_basename": "str",
                "threads": "int",
                "quiet": "bool",
                "large_index": "bool",
                "noauto": "bool",
                "packed": "bool",
                "bmax": "int",
                "bmaxdivn": "int",
                "dcv": "int",
                "offrate": "int",
                "ftabchars": "int",
                "seed": "int",
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
                    "description": "Build HISAT2 index from genome FASTA",
                    "parameters": {
                        "reference": "/data/genome.fa",
                        "index_basename": "/data/hg38_index",
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def hisat2_build(
        self,
        reference: str,
        index_basename: str,
        threads: int = 1,
        quiet: bool = False,
        large_index: bool = False,
        noauto: bool = False,
        packed: bool = False,
        bmax: int = 800,
        bmaxdivn: int = 4,
        dcv: int = 1024,
        offrate: int = 5,
        ftabchars: int = 10,
        seed: int = 0,
    ) -> dict[str, Any]:
        """
        Build HISAT2 index from genome FASTA file.

        This tool builds a HISAT2 index from a genome FASTA file, which is required
        for fast and accurate alignment of RNA-seq reads.

        Args:
            reference: Path to genome FASTA file
            index_basename: Basename for the index files
            threads: Number of threads to use
            quiet: Suppress verbose output
            large_index: Build large index (>4GB)
            noauto: Disable automatic parameter selection
            packed: Use packed representation
            bmax: Max bucket size for blockwise suffix array
            bmaxdivn: Max bucket size as divisor of ref len
            dcv: Difference-cover period
            offrate: SA sample rate
            ftabchars: Number of chars consumed in initial lookup
            seed: Random seed

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate reference file exists
        if not os.path.exists(reference):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Reference file does not exist: {reference}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Reference file not found: {reference}",
            }

        # Build command
        cmd = ["hisat2-build", reference, index_basename]

        if threads > 1:
            cmd.extend(["-p", str(threads)])
        if quiet:
            cmd.append("-q")
        if large_index:
            cmd.append("--large-index")
        if noauto:
            cmd.append("--noauto")
        if packed:
            cmd.append("--packed")
        if bmax != 800:
            cmd.extend(["--bmax", str(bmax)])
        if bmaxdivn != 4:
            cmd.extend(["--bmaxdivn", str(bmaxdivn)])
        if dcv != 1024:
            cmd.extend(["--dcv", str(dcv)])
        if offrate != 5:
            cmd.extend(["--offrate", str(offrate)])
        if ftabchars != 10:
            cmd.extend(["--ftabchars", str(ftabchars)])
        if seed != 0:
            cmd.extend(["--seed", str(seed)])

        try:
            # Execute HISAT2 index building
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # HISAT2 creates index files with various extensions
                index_extensions = [
                    ".1.ht2",
                    ".2.ht2",
                    ".3.ht2",
                    ".4.ht2",
                    ".5.ht2",
                    ".6.ht2",
                    ".7.ht2",
                    ".8.ht2",
                ]
                for ext in index_extensions:
                    index_file = f"{index_basename}{ext}"
                    if os.path.exists(index_file):
                        output_files.append(index_file)
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
                "stderr": "HISAT2 not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "HISAT2 not found in PATH",
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
            name="hisat2_align",
            description="Align RNA-seq reads to reference genome using HISAT2",
            inputs={
                "index": "str",
                "input_files": "list[str]",
                "output_file": "str",
                "unpaired": "bool",
                "mate1": "str | None",
                "mate2": "str | None",
                "threads": "int",
                "preset": "str",
                "score_min": "str",
                "no_unal": "bool",
                "no_discordant": "bool",
                "no_mixed": "bool",
                "maxins": "int",
                "minins": "int",
                "fr": "bool",
                "rf": "bool",
                "ff": "bool",
                "sam": "bool",
                "quiet": "bool",
                "seed": "int",
                "non_deterministic": "bool",
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
                    "description": "Align paired-end RNA-seq reads to genome",
                    "parameters": {
                        "index": "/data/hg38_index",
                        "input_files": ["/data/read1.fq", "/data/read2.fq"],
                        "output_file": "/data/alignment.sam",
                        "threads": 4,
                        "preset": "very-sensitive",
                    },
                }
            ],
        )
    )
    def hisat2_align(
        self,
        index: str,
        input_files: list[str],
        output_file: str,
        unpaired: bool = False,
        mate1: str | None = None,
        mate2: str | None = None,
        threads: int = 1,
        preset: str = "very-sensitive",
        score_min: str = "L,0,-0.2",
        no_unal: bool = False,
        no_discordant: bool = False,
        no_mixed: bool = False,
        maxins: int = 500,
        minins: int = 0,
        fr: bool = False,
        rf: bool = False,
        ff: bool = False,
        sam: bool = True,
        quiet: bool = False,
        seed: int = 0,
        non_deterministic: bool = False,
    ) -> dict[str, Any]:
        """
        Align RNA-seq reads to reference genome using HISAT2.

        This tool aligns RNA-seq reads to a reference genome using the HISAT2
        spliced aligner, which is optimized for RNA-seq data.

        Args:
            index: Path to HISAT2 index basename
            input_files: List of input FASTQ files
            output_file: Output SAM/BAM file
            unpaired: Input reads are unpaired
            mate1: Mate 1 FASTQ file (for paired-end)
            mate2: Mate 2 FASTQ file (for paired-end)
            threads: Number of threads to use
            preset: Alignment preset (very-fast, fast, sensitive, very-sensitive)
            score_min: Minimum score threshold
            no_unal: Suppress unpaired alignments
            no_discordant: Suppress discordant alignments
            no_mixed: Suppress mixed alignments
            maxins: Maximum insert size
            minins: Minimum insert size
            fr: First read is forward, second is reverse
            rf: First read is reverse, second is forward
            ff: Both reads are forward
            sam: Output SAM format
            quiet: Suppress verbose output
            seed: Random seed
            non_deterministic: Allow non-deterministic behavior

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate index files exist
        index_files = [
            f"{index}.1.ht2",
            f"{index}.2.ht2",
            f"{index}.3.ht2",
            f"{index}.4.ht2",
            f"{index}.5.ht2",
            f"{index}.6.ht2",
            f"{index}.7.ht2",
            f"{index}.8.ht2",
        ]

        missing_files = [f for f in index_files if not os.path.exists(f)]
        if missing_files:
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Missing index files: {', '.join(missing_files)}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Missing index files: {', '.join(missing_files)}",
            }

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

        if mate1 and not os.path.exists(mate1):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Mate 1 file does not exist: {mate1}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Mate 1 file not found: {mate1}",
            }

        if mate2 and not os.path.exists(mate2):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Mate 2 file does not exist: {mate2}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Mate 2 file not found: {mate2}",
            }

        # Build command
        cmd = ["hisat2", "-x", index]

        # Add input files
        if unpaired:
            for input_file in input_files:
                cmd.extend(["-U", input_file])
        elif mate1 and mate2:
            cmd.extend(["-1", mate1, "-2", mate2])
        else:
            for input_file in input_files:
                cmd.extend(["-U", input_file])

        # Add output
        cmd.extend(["-S", output_file])

        # Add other parameters
        cmd.extend(["--preset", preset])
        cmd.extend(["--score-min", score_min])
        if threads > 1:
            cmd.extend(["-p", str(threads)])
        if no_unal:
            cmd.append("--no-unal")
        if no_discordant:
            cmd.append("--no-discordant")
        if no_mixed:
            cmd.append("--no-mixed")
        if maxins != 500:
            cmd.extend(["--maxins", str(maxins)])
        if minins > 0:
            cmd.extend(["--minins", str(minins)])
        if fr:
            cmd.append("--fr")
        if rf:
            cmd.append("--rf")
        if ff:
            cmd.append("--ff")
        if not sam:
            cmd.append("--no-sam")
        if quiet:
            cmd.append("--quiet")
        if seed != 0:
            cmd.extend(["--seed", str(seed)])
        if non_deterministic:
            cmd.append("--non-deterministic")

        try:
            # Execute HISAT2 alignment
            result = subprocess.run(
                cmd,
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
                "stderr": "HISAT2 not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "HISAT2 not found in PATH",
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
        """Deploy HISAT2 server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-hisat2-server-{id(self)}")

            # Install HISAT2
            container.with_command(
                "bash -c 'apt-get update && apt-get install -y hisat2 && tail -f /dev/null'"
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
        """Stop HISAT2 server deployed with testcontainers."""
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
        """Get information about this HISAT2 server."""
        return {
            "name": self.name,
            "type": "hisat2",
            "version": "2.2.1",
            "description": "HISAT2 RNA-seq alignment server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
