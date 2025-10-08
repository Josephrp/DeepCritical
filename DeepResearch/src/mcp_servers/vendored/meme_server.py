"""
MEME MCP Server - Vendored BioinfoMCP server for motif discovery and sequence analysis.

This module implements a strongly-typed MCP server for MEME Suite, a collection
of tools for motif discovery and sequence analysis, using Pydantic AI patterns and testcontainers deployment.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from ...datatypes.mcp import (
    MCPAgentIntegration,
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)
from ..utils.mcp_server_base import MCPServerBase, mcp_tool


class MEMEServer(MCPServerBase):
    """MCP Server for MEME Suite motif discovery and sequence analysis tools with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="meme-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"MEME_VERSION": "5.5.4"},
                capabilities=[
                    "motif_discovery",
                    "sequence_analysis",
                    "transcription_factors",
                    "chip_seq",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def meme_motif_discovery(
        self,
        sequences: str,
        output_dir: str,
        nmotifs: int = 1,
        min_width: int = 6,
        max_width: int = 50,
        mod: str = "zoops",
        evt: float = 0.01,
        objfun: str = "classic",
        revcomp: bool = True,
        pal: bool = False,
        shuffle: bool = False,
        time: int = 7200,
        maxsize: int = 100000,
        maxiter: int = 50,
    ) -> dict[str, Any]:
        """
        Discover motifs in DNA/RNA/protein sequences using MEME.

        This tool identifies conserved motifs in a set of DNA, RNA, or protein sequences
        using expectation maximization and position weight matrices.

        Args:
            sequences: Input sequences file (FASTA format)
            output_dir: Output directory for results
            nmotifs: Maximum number of motifs to find
            min_width: Minimum motif width
            max_width: Maximum motif width
            mod: Motif model (zoops, oops, anr)
            evt: E-value threshold
            objfun: Objective function (classic, de, se)
            revcomp: Search both strands (DNA only)
            pal: Search for palindromes
            shuffle: Shuffle sequences for statistical significance
            time: Maximum runtime in seconds
            maxsize: Maximum dataset size
            maxiter: Maximum EM iterations

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input file
        seq_path = Path(sequences)
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequences file not found: {sequences}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        if nmotifs < 1:
            raise ValueError("nmotifs must be >= 1")
        if min_width < 2:
            raise ValueError("min_width must be >= 2")
        if max_width < min_width:
            raise ValueError("max_width must be >= min_width")
        if evt <= 0:
            raise ValueError("evt must be > 0")
        if time <= 0:
            raise ValueError("time must be > 0")
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if maxiter < 1:
            raise ValueError("maxiter must be >= 1")

        # Build command
        cmd = [
            "meme",
            sequences,
            "-o",
            output_dir,
            "-nmotifs",
            str(nmotifs),
            "-minw",
            str(min_width),
            "-maxw",
            str(max_width),
            "-mod",
            mod,
            "-evt",
            str(evt),
            "-objfun",
            objfun,
            "-time",
            str(time),
            "-maxsize",
            str(maxsize),
            "-maxiter",
            str(maxiter),
        ]

        if revcomp:
            cmd.append("-revcomp")

        if pal:
            cmd.append("-pal")

        if shuffle:
            cmd.append("-shuffle")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=time + 300,  # Add 5 minutes buffer
            )

            # Check for expected output files
            output_files = []
            meme_file = output_path / "meme.txt"
            if meme_file.exists():
                output_files.append(str(meme_file))

            meme_xml = output_path / "meme.xml"
            if meme_xml.exists():
                output_files.append(str(meme_xml))

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
                "error": f"MEME motif discovery failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": f"MEME motif discovery timed out after {time + 300} seconds",
            }

    @mcp_tool()
    def fimo_motif_scanning(
        self,
        sequences: str,
        motifs: str,
        output_dir: str,
        thresh: float = 1e-4,
        max_stored_scores: int = 100000,
        oc: str | None = None,
        norc: bool = False,
        bgfile: str | None = None,
        motif_pseudo: float = 0.1,
        output_pthresh: float = 1e-4,
        parse_genomic_coord: bool = False,
        text: bool = False,
        verbosity: int = 1,
    ) -> dict[str, Any]:
        """
        Scan sequences for occurrences of known motifs using FIMO.

        This tool searches for occurrences of known motifs in DNA or RNA sequences
        using position weight matrices and statistical significance testing.

        Args:
            sequences: Input sequences file (FASTA format)
            motifs: Motif file (MEME format)
            output_dir: Output directory for results
            thresh: P-value threshold for motif occurrences
            max_stored_scores: Maximum number of scores to store
            oc: Output motif occurrences file
            norc: Don't search reverse complement strand
            bgfile: Background model file
            motif_pseudo: Pseudocount for motifs
            output_pthresh: P-value threshold for output
            parse_genomic_coord: Parse genomic coordinates
            text: Output in text format
            verbosity: Verbosity level

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, success, error
        """
        # Validate input files
        seq_path = Path(sequences)
        motif_path = Path(motifs)
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequences file not found: {sequences}")
        if not motif_path.exists():
            raise FileNotFoundError(f"Motif file not found: {motifs}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        if thresh <= 0 or thresh > 1:
            raise ValueError("thresh must be between 0 and 1")
        if max_stored_scores < 1:
            raise ValueError("max_stored_scores must be >= 1")
        if motif_pseudo < 0:
            raise ValueError("motif_pseudo must be >= 0")
        if output_pthresh <= 0 or output_pthresh > 1:
            raise ValueError("output_pthresh must be between 0 and 1")
        if verbosity < 0:
            raise ValueError("verbosity must be >= 0")

        # Build command
        cmd = [
            "fimo",
            "--thresh",
            str(thresh),
            "--max-stored-scores",
            str(max_stored_scores),
            "--motif-pseudo",
            str(motif_pseudo),
            "--output-pthresh",
            str(output_pthresh),
            "--verbosity",
            str(verbosity),
            motifs,
            sequences,
        ]

        if oc:
            cmd.extend(["--oc", oc])
        else:
            cmd.extend(["--oc", output_dir])

        if norc:
            cmd.append("--norc")

        if bgfile:
            cmd.extend(["--bgfile", bgfile])

        if parse_genomic_coord:
            cmd.append("--parse-genomic-coord")

        if text:
            cmd.append("--text")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )

            # Check for expected output files
            output_files = []
            fimo_tsv = output_path / "fimo.tsv"
            if fimo_tsv.exists():
                output_files.append(str(fimo_tsv))

            fimo_xml = output_path / "fimo.xml"
            if fimo_xml.exists():
                output_files.append(str(fimo_xml))

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
                "error": f"FIMO motif scanning failed with exit code {e.returncode}: {e.stderr}",
            }
        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "success": False,
                "error": "FIMO motif scanning timed out after 3600 seconds",
            }
