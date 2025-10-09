"""
Kallisto MCP Server - Vendored BioinfoMCP server for fast RNA-seq quantification.

This module implements a strongly-typed MCP server for Kallisto, a fast and
accurate tool for quantifying abundances of transcripts from RNA-seq data,
using Pydantic AI patterns and testcontainers deployment.
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


class KallistoServer(MCPServerBase):
    """MCP Server for Kallisto RNA-seq quantification tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="kallisto-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"KALLISTO_VERSION": "0.50.1"},
                capabilities=["rna_seq", "quantification", "fast_quantification"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="kallisto_index",
            description="Build Kallisto index from transcriptome FASTA file",
            inputs={
                "fasta": "str",
                "index": "str",
                "kmer_size": "int",
                "make_unique": "bool",
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
                    "description": "Build Kallisto index from transcriptome",
                    "parameters": {
                        "fasta": "/data/transcripts.fa",
                        "index": "/data/kallisto_index",
                        "kmer_size": 31,
                    },
                }
            ],
        )
    )
    def kallisto_index(
        self,
        fasta: str,
        index: str,
        kmer_size: int = 31,
        make_unique: bool = False,
    ) -> dict[str, Any]:
        """
        Build Kallisto index from transcriptome FASTA file.

        This tool creates a Kallisto index which is required for fast and accurate
        pseudo-alignment and quantification of RNA-seq data.

        Args:
            fasta: Path to transcriptome FASTA file
            index: Path to output index file
            kmer_size: K-mer size for index building
            make_unique: Make index unique (removes duplicate sequences)

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
        if not os.path.exists(fasta):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"FASTA file does not exist: {fasta}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"FASTA file not found: {fasta}",
            }

        # Build command
        cmd = [
            "kallisto",
            "index",
            "-i",
            index,
            "-k",
            str(kmer_size),
        ]

        if make_unique:
            cmd.append("--make-unique")

        cmd.append(fasta)

        try:
            # Execute Kallisto index building
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            if os.path.exists(index):
                output_files = [index]

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
                "stderr": "Kallisto not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Kallisto not found in PATH",
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
            name="kallisto_quant",
            description="Quantify RNA-seq reads using Kallisto pseudo-alignment",
            inputs={
                "index": "str",
                "output_dir": "str",
                "single": "bool",
                "fastq1": "str | None",
                "fastq2": "str | None",
                "single_end": "str | None",
                "threads": "int",
                "fragment_length": "float",
                "sd": "float",
                "bootstrap_samples": "int",
                "seed": "int",
                "plaintext": "bool",
                "fusion": "bool",
                "single_overhang": "bool",
                "fr_stranded": "bool",
                "rf_stranded": "bool",
                "bias": "bool",
                "pseudobam": "bool",
                "genomebam": "str | None",
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
                    "description": "Quantify paired-end RNA-seq reads",
                    "parameters": {
                        "index": "/data/kallisto_index",
                        "output_dir": "/data/kallisto_quant",
                        "fastq1": "/data/sample_R1.fastq.gz",
                        "fastq2": "/data/sample_R2.fastq.gz",
                        "threads": 4,
                        "bootstrap_samples": 100,
                    },
                }
            ],
        )
    )
    def kallisto_quant(
        self,
        index: str,
        output_dir: str,
        single: bool = False,
        fastq1: str | None = None,
        fastq2: str | None = None,
        single_end: str | None = None,
        threads: int = 1,
        fragment_length: float = 200.0,
        sd: float = 20.0,
        bootstrap_samples: int = 0,
        seed: int = 42,
        plaintext: bool = False,
        fusion: bool = False,
        single_overhang: bool = False,
        fr_stranded: bool = False,
        rf_stranded: bool = False,
        bias: bool = False,
        pseudobam: bool = False,
        genomebam: str | None = None,
    ) -> dict[str, Any]:
        """
        Quantify RNA-seq reads using Kallisto pseudo-alignment.

        This tool performs fast and accurate quantification of transcript abundances
        from RNA-seq data using pseudo-alignment.

        Args:
            index: Path to Kallisto index
            output_dir: Output directory for quantification results
            single: Single-end reads
            fastq1: FASTQ file for read 1 (paired-end)
            fastq2: FASTQ file for read 2 (paired-end)
            single_end: FASTQ file for single-end reads
            threads: Number of threads to use
            fragment_length: Estimated average fragment length
            sd: Estimated standard deviation of fragment length
            bootstrap_samples: Number of bootstrap samples
            seed: Random seed
            plaintext: Output plaintext instead of HDF5
            fusion: Search for fusions
            single_overhang: Allow single-end overhang alignments
            fr_stranded: First read forward, second read reverse (stranded)
            rf_stranded: First read reverse, second read forward (stranded)
            bias: Perform sequence bias correction
            pseudobam: Output pseudoalignments in BAM format
            genomebam: Output genome alignments in BAM format

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate index exists
        if not os.path.exists(index):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Index file does not exist: {index}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Index file not found: {index}",
            }

        # Validate input files exist
        if single:
            if not single_end or not os.path.exists(single_end):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"Single-end FASTQ file does not exist: {single_end}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"Single-end FASTQ file not found: {single_end}",
                }
        else:
            if not fastq1 or not fastq2:
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": "For paired-end quantification, both fastq1 and fastq2 must be specified",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": "Missing paired-end FASTQ files",
                }
            if not os.path.exists(fastq1):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"FASTQ1 file does not exist: {fastq1}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"FASTQ1 file not found: {fastq1}",
                }
            if not os.path.exists(fastq2):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"FASTQ2 file does not exist: {fastq2}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"FASTQ2 file not found: {fastq2}",
                }

        # Build command
        cmd = [
            "kallisto",
            "quant",
            "-i",
            index,
            "-o",
            output_dir,
            "-t",
            str(threads),
            "--seed",
            str(seed),
        ]

        # Add read files
        if single:
            cmd.extend(["--single", "-l", str(fragment_length), "-s", str(sd)])
            cmd.append(single_end)
        else:
            cmd.extend([fastq1, fastq2])

        # Add options
        if bootstrap_samples > 0:
            cmd.extend(["-b", str(bootstrap_samples)])
        if plaintext:
            cmd.append("--plaintext")
        if fusion:
            cmd.append("--fusion")
        if single_overhang:
            cmd.append("--single-overhang")
        if fr_stranded:
            cmd.append("--fr-stranded")
        if rf_stranded:
            cmd.append("--rf-stranded")
        if bias:
            cmd.append("--bias")
        if pseudobam:
            cmd.append("--pseudobam")
        if genomebam:
            cmd.extend(["--genomebam", "--gtf", genomebam])

        try:
            # Execute Kallisto quantification
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # Kallisto creates various output files
                possible_outputs = [
                    os.path.join(output_dir, "abundance.tsv"),
                    os.path.join(output_dir, "abundance.h5"),
                    os.path.join(output_dir, "run_info.json"),
                ]
                for filepath in possible_outputs:
                    if os.path.exists(filepath):
                        output_files.append(filepath)
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
                "stderr": "Kallisto not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "Kallisto not found in PATH",
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
        """Deploy Kallisto server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-kallisto-server-{id(self)}")

            # Install Kallisto
            container.with_command(
                "bash -c 'pip install kallisto && tail -f /dev/null'"
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
        """Stop Kallisto server deployed with testcontainers."""
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
        """Get information about this Kallisto server."""
        return {
            "name": self.name,
            "type": "kallisto",
            "version": "0.50.1",
            "description": "Kallisto RNA-seq quantification server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
