"""
BWA MCP Server - Vendored BioinfoMCP server for DNA sequence alignment.

This module implements a strongly-typed MCP server for BWA (Burrows-Wheeler Aligner),
a fast and accurate short read aligner for DNA sequencing data, using Pydantic AI
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
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)


class BWAServer(MCPServerBase):
    """MCP Server for BWA DNA sequence alignment tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="bwa-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"BWA_VERSION": "0.7.17"},
                capabilities=["dna_alignment", "short_read_alignment", "genomics"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="bwa_index",
            description="Build BWA index from reference genome FASTA file",
            inputs={
                "reference": "str",
                "algorithm": "str",
                "prefix": "str",
            },
            outputs={
                "command_executed": "str",
                "stdout": "str",
                "stderr": "str",
                "output_files": "List[str]",
                "exit_code": "int",
            },
            server_type=MCPServerType.CUSTOM,
            examples=[
                {
                    "description": "Build BWA index from reference genome",
                    "parameters": {
                        "reference": "/data/genome.fa",
                        "algorithm": "bwtsw",
                        "prefix": "/data/hg38",
                    },
                }
            ],
        )
    )
    def bwa_index(
        self,
        reference: str,
        algorithm: str = "bwtsw",
        prefix: str = "",
    ) -> dict[str, Any]:
        """
        Build BWA index from reference genome FASTA file.

        This tool builds a BWA index from a reference genome FASTA file,
        which is required for fast and accurate alignment of DNA sequencing reads.

        Args:
            reference: Path to reference genome FASTA file
            algorithm: Indexing algorithm (bwtsw, is)
            prefix: Prefix for index files (optional, defaults to reference prefix)

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
        cmd = ["bwa", "index", "-a", algorithm]

        if prefix:
            cmd.extend(["-p", prefix])
            index_prefix = prefix
        else:
            # Use reference filename as prefix
            index_prefix = reference.rsplit(".", 1)[0]

        cmd.append(reference)

        try:
            # Execute BWA index building
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # BWA creates index files with various extensions
                index_extensions = [".amb", ".ann", ".bwt", ".pac", ".sa"]
                for ext in index_extensions:
                    index_file = f"{index_prefix}{ext}"
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
                "stderr": "BWA not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "BWA not found in PATH",
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
            name="bwa_mem",
            description="Align DNA sequencing reads using BWA-MEM algorithm",
            inputs={
                "index": "str",
                "read1": "str",
                "read2": "Optional[str]",
                "output_file": "str",
                "threads": "int",
                "min_seed_length": "int",
                "band_width": "int",
                "off_diag": "int",
                "no_rescue": "bool",
                "skip_mate_rescue": "bool",
                "skip_pairing": "bool",
                "match_score": "int",
                "mismatch_penalty": "int",
                "gap_open_penalty": "int",
                "gap_extension_penalty": "int",
                "clipping_penalty": "int",
                "unpaired_read_penalty": "int",
                "score_threshold": "int",
                "split_factor": "float",
                "split_width": "int",
                "h": "bool",
                "a": "bool",
                "c": "bool",
                "mark_secondary_splits": "bool",
                "u": "bool",
                "r": "str",
            },
            outputs={
                "command_executed": "str",
                "stdout": "str",
                "stderr": "str",
                "output_files": "List[str]",
                "exit_code": "int",
            },
            server_type=MCPServerType.CUSTOM,
            examples=[
                {
                    "description": "Align paired-end DNA reads to reference",
                    "parameters": {
                        "index": "/data/hg38",
                        "read1": "/data/read1.fq",
                        "read2": "/data/read2.fq",
                        "output_file": "/data/alignment.sam",
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def bwa_mem(
        self,
        index: str,
        read1: str,
        read2: str | None = None,
        output_file: str = "",
        threads: int = 1,
        min_seed_length: int = 19,
        band_width: int = 100,
        off_diag: int = 100,
        no_rescue: bool = False,
        skip_mate_rescue: bool = False,
        skip_pairing: bool = False,
        match_score: int = 1,
        mismatch_penalty: int = 4,
        gap_open_penalty: int = 6,
        gap_extension_penalty: int = 1,
        clipping_penalty: int = 5,
        unpaired_read_penalty: int = 17,
        score_threshold: int = 30,
        split_factor: float = 1.5,
        split_width: int = 16,
        h: bool = False,
        a: bool = False,
        c: bool = False,
        mark_secondary_splits: bool = False,
        u: bool = False,
        r: str = "",
    ) -> dict[str, Any]:
        """
        Align DNA sequencing reads using BWA-MEM algorithm.

        This tool aligns DNA sequencing reads to a reference genome using BWA-MEM,
        which is optimized for high-throughput sequencing data.

        Args:
            index: Path to BWA index prefix
            read1: Path to first read file
            read2: Path to second read file (optional, for paired-end)
            output_file: Output SAM file
            threads: Number of threads to use
            min_seed_length: Minimum seed length
            band_width: Band width for banded alignment
            off_diag: Off-diagonal X-dropoff
            no_rescue: Skip mate rescue
            skip_mate_rescue: Skip mate rescue
            skip_pairing: Skip pairing
            match_score: Match score
            mismatch_penalty: Mismatch penalty
            gap_open_penalty: Gap open penalty
            gap_extension_penalty: Gap extension penalty
            clipping_penalty: Clipping penalty
            unpaired_read_penalty: Unpaired read penalty
            score_threshold: Minimum score to output
            split_factor: Split factor
            split_width: Split width
            h: Use hard clipping
            a: Output all alignments
            c: Append FASTA/FASTQ comment to SAM output
            mark_secondary_splits: Mark shorter split hits as secondary
            u: Output unmapped reads
            r: Read group header line

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate index files exist
        index_files = [
            f"{index}.amb",
            f"{index}.ann",
            f"{index}.bwt",
            f"{index}.pac",
            f"{index}.sa",
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
        if not os.path.exists(read1):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Read 1 file does not exist: {read1}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Read 1 file not found: {read1}",
            }

        if read2 and not os.path.exists(read2):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Read 2 file does not exist: {read2}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Read 2 file not found: {read2}",
            }

        # Build command
        cmd = ["bwa", "mem", "-t", str(threads)]

        # Add scoring parameters
        cmd.extend(["-k", str(min_seed_length)])
        cmd.extend(["-w", str(band_width)])
        cmd.extend(["-d", str(off_diag)])
        cmd.extend(["-r", str(split_factor)])
        cmd.extend(["-c", str(split_width)])
        cmd.extend(["-A", str(match_score)])
        cmd.extend(["-B", str(mismatch_penalty)])
        cmd.extend(["-O", str(gap_open_penalty)])
        cmd.extend(["-E", str(gap_extension_penalty)])
        cmd.extend(["-L", str(clipping_penalty)])
        cmd.extend(["-U", str(unpaired_read_penalty)])
        cmd.extend(["-T", str(score_threshold)])

        # Add boolean flags
        if no_rescue:
            cmd.append("-P")
        if skip_mate_rescue:
            cmd.append("-S")
        if skip_pairing:
            cmd.append("-P")
        if h:
            cmd.append("-H")
        if a:
            cmd.append("-a")
        if c:
            cmd.append("-C")
        if mark_secondary_splits:
            cmd.append("-L")
        if u:
            cmd.append("-U")

        # Add read group
        if r:
            cmd.extend(["-R", r])

        # Add index and reads
        cmd.append(index)

        if read2:
            cmd.extend([read1, read2])
        else:
            cmd.append(read1)

        try:
            # Execute BWA-MEM alignment with output redirection
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
                "stderr": "BWA not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "BWA not found in PATH",
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
            name="bwa_aln",
            description="Align DNA sequencing reads using BWA-ALN algorithm",
            inputs={
                "index": "str",
                "read1": "str",
                "read2": "Optional[str]",
                "output_file": "str",
                "threads": "int",
                "min_seed_length": "int",
                "band_width": "int",
                "off_diag": "int",
                "no_rescue": "bool",
                "skip_mate_rescue": "bool",
                "skip_pairing": "bool",
                "match_score": "int",
                "mismatch_penalty": "int",
                "gap_open_penalty": "int",
                "gap_extension_penalty": "int",
                "clipping_penalty": "int",
                "unpaired_read_penalty": "int",
                "score_threshold": "int",
                "split_factor": "float",
                "split_width": "int",
            },
            outputs={
                "command_executed": "str",
                "stdout": "str",
                "stderr": "str",
                "output_files": "List[str]",
                "exit_code": "int",
            },
            server_type=MCPServerType.CUSTOM,
            examples=[
                {
                    "description": "Align paired-end DNA reads using BWA-ALN",
                    "parameters": {
                        "index": "/data/hg38",
                        "read1": "/data/read1.fq",
                        "read2": "/data/read2.fq",
                        "output_file": "/data/alignment.sai",
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def bwa_aln(
        self,
        index: str,
        read1: str,
        read2: str | None = None,
        output_file: str = "",
        threads: int = 1,
        min_seed_length: int = 32,
        band_width: int = 100,
        off_diag: int = 100,
        no_rescue: bool = False,
        skip_mate_rescue: bool = False,
        skip_pairing: bool = False,
        match_score: int = 1,
        mismatch_penalty: int = 3,
        gap_open_penalty: int = 11,
        gap_extension_penalty: int = 4,
        clipping_penalty: int = 5,
        unpaired_read_penalty: int = 9,
        score_threshold: int = 30,
        split_factor: float = 1.5,
        split_width: int = 16,
    ) -> dict[str, Any]:
        """
        Align DNA sequencing reads using BWA-ALN algorithm.

        This tool aligns DNA sequencing reads using BWA-ALN, which is optimized
        for reads up to 100bp. For longer reads, use BWA-MEM instead.

        Args:
            index: Path to BWA index prefix
            read1: Path to first read file
            read2: Path to second read file (optional, for paired-end)
            output_file: Output SAI file
            threads: Number of threads to use
            min_seed_length: Minimum seed length
            band_width: Band width for banded alignment
            off_diag: Off-diagonal X-dropoff
            no_rescue: Skip mate rescue
            skip_mate_rescue: Skip mate rescue
            skip_pairing: Skip pairing
            match_score: Match score
            mismatch_penalty: Mismatch penalty
            gap_open_penalty: Gap open penalty
            gap_extension_penalty: Gap extension penalty
            clipping_penalty: Clipping penalty
            unpaired_read_penalty: Unpaired read penalty
            score_threshold: Minimum score to output
            split_factor: Split factor
            split_width: Split width

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate index files exist
        index_files = [
            f"{index}.amb",
            f"{index}.ann",
            f"{index}.bwt",
            f"{index}.pac",
            f"{index}.sa",
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
        if not os.path.exists(read1):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Read 1 file does not exist: {read1}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Read 1 file not found: {read1}",
            }

        if read2 and not os.path.exists(read2):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Read 2 file does not exist: {read2}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Read 2 file not found: {read2}",
            }

        # Build command
        cmd = ["bwa", "aln", "-t", str(threads)]

        # Add scoring parameters
        cmd.extend(["-k", str(min_seed_length)])
        cmd.extend(["-w", str(band_width)])
        cmd.extend(["-d", str(off_diag)])
        cmd.extend(["-r", str(split_factor)])
        cmd.extend(["-c", str(split_width)])
        cmd.extend(["-A", str(match_score)])
        cmd.extend(["-B", str(mismatch_penalty)])
        cmd.extend(["-O", str(gap_open_penalty)])
        cmd.extend(["-E", str(gap_extension_penalty)])
        cmd.extend(["-L", str(clipping_penalty)])
        cmd.extend(["-U", str(unpaired_read_penalty)])
        cmd.extend(["-T", str(score_threshold)])

        # Add boolean flags
        if no_rescue:
            cmd.append("-P")
        if skip_mate_rescue:
            cmd.append("-S")
        if skip_pairing:
            cmd.append("-P")

        # Add index and reads
        cmd.append(index)
        cmd.append(read1)

        try:
            # Execute BWA-ALN alignment with output redirection
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
                "stderr": "BWA not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "BWA not found in PATH",
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
        """Deploy BWA server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-bwa-server-{id(self)}")

            # Install BWA
            container.with_command(
                "bash -c 'apt-get update && apt-get install -y wget bwa && tail -f /dev/null'"
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
        """Stop BWA server deployed with testcontainers."""
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
        """Get information about this BWA server."""
        return {
            "name": self.name,
            "type": "bwa",
            "version": "0.7.17",
            "description": "BWA DNA sequence alignment server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
