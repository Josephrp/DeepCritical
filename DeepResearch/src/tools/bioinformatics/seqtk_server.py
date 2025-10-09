"""
Seqtk MCP Server - Vendored BioinfoMCP server for FASTA/Q processing.

This module implements a strongly-typed MCP server for Seqtk, a fast and lightweight
tool for processing FASTA/Q files, using Pydantic AI patterns and testcontainers deployment.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ...datatypes.bioinformatics_mcp import MCPServerBase, mcp_tool
from ...datatypes.mcp import (
    MCPAgentIntegration,
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)


class SeqtkServer(MCPServerBase):
    """MCP Server for Seqtk FASTA/Q processing tools with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="seqtk-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"SEQTK_VERSION": "1.3"},
                capabilities=[
                    "sequence_processing",
                    "fasta_manipulation",
                    "fastq_manipulation",
                    "quality_control",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def seqtk_subseq(
        self,
        input_file: str,
        region_file: str,
        output_file: str,
        tab_indexed: bool = False,
        uppercase: bool = False,
        mask_lowercase: bool = False,
        reverse_complement: bool = False,
        name_only: bool = False,
    ) -> dict[str, Any]:
        """
        Extract subsequences from FASTA/Q files using Seqtk.

        This tool extracts specific sequences or subsequences from FASTA/Q files
        based on sequence names or genomic coordinates.

        Args:
            input_file: Input FASTA/Q file
            region_file: File containing regions/sequence names to extract
            output_file: Output FASTA/Q file
            tab_indexed: Input is tab-delimited (name\tseq format)
            uppercase: Convert sequences to uppercase
            mask_lowercase: Mask lowercase letters with 'N'
            reverse_complement: Output reverse complement
            name_only: Output sequence names only

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        input_path = Path(input_file)
        region_path = Path(region_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not region_path.exists():
            raise FileNotFoundError(f"Region file not found: {region_file}")

        # Build command
        cmd = ["seqtk", "subseq", input_file, region_file]

        if tab_indexed:
            cmd.append("-t")

        if uppercase:
            cmd.append("-U")

        if mask_lowercase:
            cmd.append("-l")

        if reverse_complement:
            cmd.append("-r")

        if name_only:
            cmd.append("-n")

        # Redirect output to file
        cmd.extend([">", output_file])

        try:
            # Use shell=True to handle output redirection
            result = subprocess.run(
                " ".join(cmd),
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )

            output_files = []
            if Path(output_file).exists():
                output_files.append(output_file)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "success": False,
                "error": f"Seqtk subseq failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Seqtk subseq timed out after 600 seconds",
            }

    @mcp_tool()
    def seqtk_sample(
        self,
        input_file: str,
        fraction: float,
        output_file: str,
        seed: int | None = None,
        two_pass: bool = False,
    ) -> dict[str, Any]:
        """
        Randomly sample sequences from FASTA/Q files using Seqtk.

        This tool randomly samples a fraction or specific number of sequences
        from FASTA/Q files for downstream analysis.

        Args:
            input_file: Input FASTA/Q file
            fraction: Fraction of sequences to sample (0.0-1.0) or number (>1)
            output_file: Output FASTA/Q file
            seed: Random seed for reproducible sampling
            two_pass: Use two-pass algorithm for exact sampling

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Validate fraction
        if fraction <= 0:
            raise ValueError("fraction must be > 0")
        if fraction > 1 and fraction != int(fraction):
            raise ValueError("fraction > 1 must be an integer")

        # Build command
        cmd = ["seqtk", "sample", "-s100"]

        if seed is not None:
            cmd.extend(["-s", str(seed)])

        if two_pass:
            cmd.append("-2")

        cmd.extend([input_file, str(fraction)])

        # Redirect output to file
        full_cmd = " ".join(cmd) + f" > {output_file}"

        try:
            # Use shell=True to handle output redirection
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )

            output_files = []
            if Path(output_file).exists():
                output_files.append(output_file)

            return {
                "command_executed": full_cmd,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": full_cmd,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "success": False,
                "error": f"Seqtk sample failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": full_cmd,
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Seqtk sample timed out after 600 seconds",
            }

    @mcp_tool()
    def seqtk_mergepe(
        self,
        read1_file: str,
        read2_file: str,
        output_file: str,
    ) -> dict[str, Any]:
        """
        Merge paired-end FASTQ files into interleaved format using Seqtk.

        This tool interleaves paired-end FASTQ files for tools that require
        interleaved input format.

        Args:
            read1_file: First read FASTQ file
            read2_file: Second read FASTQ file
            output_file: Output interleaved FASTQ file

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        read1_path = Path(read1_file)
        read2_path = Path(read2_file)
        if not read1_path.exists():
            raise FileNotFoundError(f"Read1 file not found: {read1_file}")
        if not read2_path.exists():
            raise FileNotFoundError(f"Read2 file not found: {read2_file}")

        # Build command
        cmd = ["seqtk", "mergepe", read1_file, read2_file]

        # Redirect output to file
        full_cmd = " ".join(cmd) + f" > {output_file}"

        try:
            # Use shell=True to handle output redirection
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )

            output_files = []
            if Path(output_file).exists():
                output_files.append(output_file)

            return {
                "command_executed": full_cmd,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": full_cmd,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "success": False,
                "error": f"Seqtk mergepe failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": full_cmd,
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Seqtk mergepe timed out after 600 seconds",
            }

    @mcp_tool()
    def seqtk_comp(
        self,
        input_file: str,
        output_file: str | None = None,
    ) -> dict[str, Any]:
        """
        Count base composition of FASTA/Q files using Seqtk.

        This tool provides statistics on nucleotide composition and quality
        scores in FASTA/Q files.

        Args:
            input_file: Input FASTA/Q file
            output_file: Optional output file (default: stdout)

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Build command
        cmd = ["seqtk", "comp", input_file]

        if output_file:
            # Redirect output to file
            full_cmd = " ".join(cmd) + f" > {output_file}"
            shell_cmd = full_cmd
        else:
            full_cmd = " ".join(cmd)
            shell_cmd = full_cmd

        try:
            # Use shell=True to handle output redirection
            result = subprocess.run(
                shell_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )

            output_files = []
            if output_file and Path(output_file).exists():
                output_files.append(output_file)

            return {
                "command_executed": full_cmd,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": full_cmd,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "success": False,
                "error": f"Seqtk comp failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": full_cmd,
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "Seqtk comp timed out after 600 seconds",
            }
