"""
Bowtie2 MCP Server - Vendored BioinfoMCP server for sequence alignment.

This module implements a strongly-typed MCP server for Bowtie2, an ultrafast
and memory-efficient tool for aligning sequencing reads to long reference sequences.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Note: In a real implementation, you would import mcp here
# from mcp import tool
from ...datatypes.mcp import (
    MCPAgentIntegration,
    MCPServerConfig,
    MCPServerDeployment,
    MCPServerStatus,
    MCPServerType,
    MCPToolSpec,
)
from ..utils.mcp_server_base import MCPServerBase, mcp_tool


class Bowtie2Server(MCPServerBase):
    """MCP Server for Bowtie2 sequence alignment tool."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="bowtie2-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"BOWTIE2_VERSION": "2.5.1"},
                capabilities=["sequence_alignment", "read_mapping", "genome_alignment"],
            )
        super().__init__(config)

    @mcp_tool()
    def bowtie2_align(
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
        met_file: str | None = None,
        met_stderr: bool = False,
        met: int = 1,
        seed: int = 0,
        non_deterministic: bool = False,
    ) -> dict[str, Any]:
        """
        Align sequencing reads to a reference genome using Bowtie2.

        Args:
            index: Bowtie2 index basename
            input_files: List of input FASTQ files (for unpaired reads)
            output_file: Output SAM/BAM file
            unpaired: Input files are unpaired reads
            mate1: File with #1 mates (for paired-end)
            mate2: File with #2 mates (for paired-end)
            threads: Number of threads to use
            preset: Alignment preset (very-fast, fast, sensitive, very-sensitive)
            score_min: Minimum score threshold
            no_unal: Suppress unpaired alignments
            no_discordant: Suppress discordant alignments
            no_mixed: Suppress mixed alignments
            maxins: Maximum insert size
            minins: Minimum insert size
            fr: Expect -> <- orientation for paired reads
            rf: Expect <- -> orientation for paired reads
            ff: Expect -> -> orientation for paired reads
            sam: Output in SAM format
            quiet: Suppress progress messages
            met_file: Send metrics to file
            met_stderr: Send metrics to stderr
            met: Report interval
            seed: Random seed
            non_deterministic: Allow non-deterministic alignments

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate index exists
        for ext in [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]:
            index_file = f"{index}{ext}"
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Bowtie2 index file not found: {index_file}")

        # Validate input files
        if unpaired:
            for input_file in input_files:
                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"Input file not found: {input_file}")
        else:
            if not mate1 or not mate2:
                raise ValueError(
                    "For paired-end alignment, both mate1 and mate2 must be specified"
                )
            if not os.path.exists(mate1):
                raise FileNotFoundError(f"Mate1 file not found: {mate1}")
            if not os.path.exists(mate2):
                raise FileNotFoundError(f"Mate2 file not found: {mate2}")

        # Build command
        cmd = ["bowtie2"]

        # Add options
        if threads > 1:
            cmd.extend(["-p", str(threads)])
        if preset != "very-sensitive":
            cmd.extend(["--preset", preset])
        if score_min != "L,0,-0.2":
            cmd.extend(["--score-min", score_min])
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
        elif rf:
            cmd.append("--rf")
        elif ff:
            cmd.append("--ff")
        if not sam:
            cmd.append("--no-sam")
        if quiet:
            cmd.append("--quiet")
        if met_file:
            cmd.extend(["--met-file", met_file])
        if met_stderr:
            cmd.append("--met-stderr")
        if met != 1:
            cmd.extend(["--met", str(met)])
        if seed != 0:
            cmd.extend(["--seed", str(seed)])
        if non_deterministic:
            cmd.append("--non-deterministic")

        # Add index
        cmd.extend(["-x", index])

        # Add reads
        if unpaired:
            for input_file in input_files:
                cmd.extend(["-U", input_file])
        else:
            cmd.extend(["-1", mate1, "-2", mate2])

        # Redirect output
        cmd.extend(["-S", output_file])

        # Execute command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": [output_file],
                "exit_code": result.returncode,
                "success": True,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "exit_code": e.returncode,
                "success": False,
                "error": f"bowtie2 alignment failed: {e}",
            }

        except Exception as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": str(e),
            }

    @mcp_tool()
    def bowtie2_build(
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
        cutoffs: str | None = None,
    ) -> dict[str, Any]:
        """
        Build a Bowtie2 index from a reference genome.

        Args:
            reference: Reference genome FASTA file
            index_basename: Basename for the index files
            threads: Number of threads to use
            quiet: Suppress progress messages
            large_index: Force large index
            noauto: Disable automatic parameter selection
            packed: Use packed index format
            bmax: Max bucket size
            bmaxdivn: Bmax divided by n
            dcv: Difference-cover period
            offrate: SA sample rate
            ftabchars: Number of chars in ftab
            seed: Random seed
            cutoffs: Reference cutoffs

        Returns:
            Dictionary containing command executed, stdout, stderr, and output files
        """
        # Validate reference file exists
        if not os.path.exists(reference):
            raise FileNotFoundError(f"Reference file not found: {reference}")

        # Build command
        cmd = ["bowtie2-build"]

        # Add options
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
        if cutoffs:
            cmd.extend(["--cutoffs", cutoffs])

        # Add reference and index basename
        cmd.extend([reference, index_basename])

        # Execute command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Find output files
            output_files = []
            for ext in [
                ".1.bt2",
                ".2.bt2",
                ".3.bt2",
                ".4.bt2",
                ".rev.1.bt2",
                ".rev.2.bt2",
            ]:
                index_file = f"{index_basename}{ext}"
                if os.path.exists(index_file):
                    output_files.append(index_file)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "output_files": [],
                "exit_code": e.returncode,
                "success": False,
                "error": f"bowtie2-build failed: {e}",
            }

        except Exception as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": str(e),
            }

    @mcp_tool()
    def bowtie2_inspect(
        self,
        index: str,
        summary: bool = False,
        names: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Inspect a Bowtie2 index.

        Args:
            index: Bowtie2 index basename
            summary: Print summary of index
            names: Print sequence names
            verbose: Verbose output

        Returns:
            Dictionary containing command executed, stdout, stderr, and index information
        """
        # Validate index exists
        for ext in [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]:
            index_file = f"{index}{ext}"
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"Bowtie2 index file not found: {index_file}")

        # Build command
        cmd = ["bowtie2-inspect"]

        # Add options
        if summary:
            cmd.append("-s")
        if names:
            cmd.append("-n")
        if verbose:
            cmd.append("-v")

        # Add index
        cmd.append(index)

        # Execute command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "success": True,
                "index_information": result.stdout,
            }

        except subprocess.CalledProcessError as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "exit_code": e.returncode,
                "success": False,
                "error": f"bowtie2-inspect failed: {e}",
            }

        except Exception as e:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "success": False,
                "error": str(e),
            }

    async def deploy_with_testcontainers(self) -> MCPServerDeployment:
        """Deploy Bowtie2 server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-bowtie2-server-{id(self)}")

            # Install Bowtie2
            container.with_command("bash -c 'pip install bowtie2 && tail -f /dev/null'")

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
        """Stop Bowtie2 server deployed with testcontainers."""
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
        """Get information about this Bowtie2 server."""
        return {
            "name": self.name,
            "type": "bowtie2",
            "version": "2.5.1",
            "description": "Bowtie2 sequence alignment server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }


# Create server instance
bowtie2_server = Bowtie2Server()
