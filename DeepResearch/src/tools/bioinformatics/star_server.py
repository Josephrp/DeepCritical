"""
STAR MCP Server - Vendored BioinfoMCP server for RNA-seq alignment.

This module implements a strongly-typed MCP server for STAR, a popular
spliced read aligner for RNA-seq data, using Pydantic AI patterns and
testcontainers deployment.
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


class STARServer(MCPServerBase):
    """MCP Server for STAR RNA-seq alignment tool with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="star-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"STAR_VERSION": "2.7.10b"},
                capabilities=["rna_seq", "alignment", "spliced_alignment"],
            )
        super().__init__(config)

    @mcp_tool(
        MCPToolSpec(
            name="star_generate_genome",
            description="Generate STAR genome index from genome FASTA and GTF files",
            inputs={
                "genome_dir": "str",
                "genome_fasta_files": "list[str]",
                "sjdb_gtf_file": "str | None",
                "sjdb_overhang": "int",
                "genome_sa_index_n_bases": "int",
                "genome_chr_bin_n_bits": "int",
                "genome_sa_sparse_d": "int",
                "threads": "int",
                "limit_genome_generate_ram": "str",
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
                    "description": "Generate STAR genome index for human genome",
                    "parameters": {
                        "genome_dir": "/data/star_index",
                        "genome_fasta_files": ["/data/genome.fa"],
                        "sjdb_gtf_file": "/data/genes.gtf",
                        "sjdb_overhang": 149,
                        "threads": 4,
                    },
                }
            ],
        )
    )
    def star_generate_genome(
        self,
        genome_dir: str,
        genome_fasta_files: list[str],
        sjdb_gtf_file: str | None = None,
        sjdb_overhang: int = 100,
        genome_sa_index_n_bases: int = 14,
        genome_chr_bin_n_bits: int = 18,
        genome_sa_sparse_d: int = 1,
        threads: int = 1,
        limit_genome_generate_ram: str = "31000000000",
    ) -> dict[str, Any]:
        """
        Generate STAR genome index from genome FASTA and GTF files.

        This tool creates a STAR genome index which is required for fast and accurate
        alignment of RNA-seq reads using the STAR aligner.

        Args:
            genome_dir: Directory to store the genome index
            genome_fasta_files: List of genome FASTA files
            sjdb_gtf_file: GTF file with gene annotations
            sjdb_overhang: Read length - 1 (for paired-end reads, use read length - 1)
            genome_sa_index_n_bases: Length (bases) of the SA pre-indexing string
            genome_chr_bin_n_bits: Number of bits for genome chromosome bins
            genome_sa_sparse_d: Suffix array sparsity
            threads: Number of threads to use
            limit_genome_generate_ram: Maximum RAM for genome generation

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        for fasta_file in genome_fasta_files:
            if not os.path.exists(fasta_file):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"Genome FASTA file does not exist: {fasta_file}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"Genome FASTA file not found: {fasta_file}",
                }

        if sjdb_gtf_file and not os.path.exists(sjdb_gtf_file):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"GTF file does not exist: {sjdb_gtf_file}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"GTF file not found: {sjdb_gtf_file}",
            }

        # Build command
        cmd = ["STAR", "--runMode", "genomeGenerate", "--genomeDir", genome_dir]

        # Add genome FASTA files
        cmd.extend(["--genomeFastaFiles"] + genome_fasta_files)

        if sjdb_gtf_file:
            cmd.extend(["--sjdbGTFfile", sjdb_gtf_file])

        cmd.extend(
            [
                "--sjdbOverhang",
                str(sjdb_overhang),
                "--genomeSAindexNbases",
                str(genome_sa_index_n_bases),
                "--genomeChrBinNbits",
                str(genome_chr_bin_n_bits),
                "--genomeSASparseD",
                str(genome_sa_sparse_d),
                "--runThreadN",
                str(threads),
                "--limitGenomeGenerateRAM",
                limit_genome_generate_ram,
            ]
        )

        try:
            # Execute STAR genome generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # STAR creates various index files
                index_files = [
                    "Genome",
                    "SA",
                    "SAindex",
                    "chrLength.txt",
                    "chrName.txt",
                    "chrNameLength.txt",
                    "chrStart.txt",
                    "exonGeTrInfo.tab",
                    "exonInfo.tab",
                    "geneInfo.tab",
                    "genomeParameters.txt",
                    "sjdbInfo.txt",
                    "sjdbList.fromGTF.out.tab",
                    "sjdbList.out.tab",
                    "transcriptInfo.tab",
                ]
                for filename in index_files:
                    filepath = os.path.join(genome_dir, filename)
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
                "stderr": "STAR not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "STAR not found in PATH",
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
            name="star_align_reads",
            description="Align RNA-seq reads to reference genome using STAR",
            inputs={
                "genome_dir": "str",
                "read_files_in": "list[str]",
                "out_file_name_prefix": "str",
                "run_thread_n": "int",
                "out_sam_type": "str",
                "out_sam_mode": "str",
                "quant_mode": "str",
                "read_files_command": "str | None",
                "out_filter_multimap_nmax": "int",
                "out_filter_mismatch_nmax": "int",
                "align_intron_min": "int",
                "align_intron_max": "int",
                "align_mates_gap_max": "int",
                "chim_segment_min": "int",
                "chim_junction_overhang_min": "int",
                "twopass_mode": "str",
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
                    "description": "Align paired-end RNA-seq reads",
                    "parameters": {
                        "genome_dir": "/data/star_index",
                        "read_files_in": ["/data/sample1.fastq", "/data/sample2.fastq"],
                        "out_file_name_prefix": "/results/sample_",
                        "run_thread_n": 4,
                        "quant_mode": "TranscriptomeSAM",
                    },
                }
            ],
        )
    )
    def star_align_reads(
        self,
        genome_dir: str,
        read_files_in: list[str],
        out_file_name_prefix: str,
        run_thread_n: int = 1,
        out_sam_type: str = "BAM SortedByCoordinate",
        out_sam_mode: str = "Full",
        quant_mode: str = "GeneCounts",
        read_files_command: str | None = None,
        out_filter_multimap_nmax: int = 20,
        out_filter_mismatch_nmax: int = 999,
        align_intron_min: int = 21,
        align_intron_max: int = 0,
        align_mates_gap_max: int = 0,
        chim_segment_min: int = 0,
        chim_junction_overhang_min: int = 20,
        twopass_mode: str = "Basic",
    ) -> dict[str, Any]:
        """
        Align RNA-seq reads to reference genome using STAR.

        This tool aligns RNA-seq reads to a reference genome using the STAR spliced
        aligner, which is optimized for RNA-seq data and provides high accuracy.

        Args:
            genome_dir: Directory containing STAR genome index
            read_files_in: List of input FASTQ files
            out_file_name_prefix: Prefix for output files
            run_thread_n: Number of threads to use
            out_sam_type: Output SAM type (SAM, BAM, etc.)
            out_sam_mode: Output SAM mode (Full, None)
            quant_mode: Quantification mode (GeneCounts, TranscriptomeSAM)
            read_files_command: Command to process input files
            out_filter_multimap_nmax: Maximum number of multiple alignments
            out_filter_mismatch_nmax: Maximum number of mismatches
            align_intron_min: Minimum intron length
            align_intron_max: Maximum intron length (0 = no limit)
            align_mates_gap_max: Maximum gap between mates
            chim_segment_min: Minimum chimeric segment length
            chim_junction_overhang_min: Minimum chimeric junction overhang
            twopass_mode: Two-pass mapping mode

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate genome directory exists
        if not os.path.exists(genome_dir):
            return {
                "command_executed": "",
                "stdout": "",
                "stderr": f"Genome directory does not exist: {genome_dir}",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": f"Genome directory not found: {genome_dir}",
            }

        # Validate input files exist
        for read_file in read_files_in:
            if not os.path.exists(read_file):
                return {
                    "command_executed": "",
                    "stdout": "",
                    "stderr": f"Read file does not exist: {read_file}",
                    "output_files": [],
                    "exit_code": -1,
                    "success": False,
                    "error": f"Read file not found: {read_file}",
                }

        # Build command
        cmd = ["STAR", "--genomeDir", genome_dir]

        # Add input read files
        cmd.extend(["--readFilesIn"] + read_files_in)

        # Add output prefix
        cmd.extend(["--outFileNamePrefix", out_file_name_prefix])

        # Add other parameters
        cmd.extend(
            [
                "--runThreadN",
                str(run_thread_n),
                "--outSAMtype",
                out_sam_type,
                "--outSAMmode",
                out_sam_mode,
                "--quantMode",
                quant_mode,
                "--outFilterMultimapNmax",
                str(out_filter_multimap_nmax),
                "--outFilterMismatchNmax",
                str(out_filter_mismatch_nmax),
                "--alignIntronMin",
                str(align_intron_min),
                "--alignIntronMax",
                str(align_intron_max),
                "--alignMatesGapMax",
                str(align_mates_gap_max),
                "--chimSegmentMin",
                str(chim_segment_min),
                "--chimJunctionOverhangMin",
                str(chim_junction_overhang_min),
                "--twopassMode",
                twopass_mode,
            ]
        )

        if read_files_command:
            cmd.extend(["--readFilesCommand", read_files_command])

        try:
            # Execute STAR alignment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get output files
            output_files = []
            try:
                # STAR creates various output files
                possible_outputs = [
                    f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam",
                    f"{out_file_name_prefix}ReadsPerGene.out.tab",
                    f"{out_file_name_prefix}Log.final.out",
                    f"{out_file_name_prefix}Log.out",
                    f"{out_file_name_prefix}Log.progress.out",
                    f"{out_file_name_prefix}SJ.out.tab",
                    f"{out_file_name_prefix}Chimeric.out.junction",
                    f"{out_file_name_prefix}Chimeric.out.sam",
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
                "stderr": "STAR not found in PATH",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "STAR not found in PATH",
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
        """Deploy STAR server using testcontainers."""
        try:
            from testcontainers.core.container import DockerContainer

            # Create container
            container = DockerContainer("python:3.11-slim")
            container.with_name(f"mcp-star-server-{id(self)}")

            # Install STAR
            container.with_command(
                "bash -c 'apt-get update && apt-get install -y star && tail -f /dev/null'"
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
        """Stop STAR server deployed with testcontainers."""
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
        """Get information about this STAR server."""
        return {
            "name": self.name,
            "type": "star",
            "version": "2.7.10b",
            "description": "STAR RNA-seq alignment server",
            "tools": self.list_tools(),
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": "running" if self.container_id else "stopped",
        }
