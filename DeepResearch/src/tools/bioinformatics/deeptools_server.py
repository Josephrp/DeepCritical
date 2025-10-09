"""
Deeptools MCP Server - Vendored BioinfoMCP server for deep sequencing data analysis and visualization.

This module implements a strongly-typed MCP server for Deeptools, a comprehensive suite
of tools for the analysis and visualization of deep sequencing data, particularly useful
for ChIP-seq and RNA-seq data analysis using Pydantic AI patterns and testcontainers deployment.
"""

from __future__ import annotations

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


class DeeptoolsServer(MCPServerBase):
    """MCP Server for Deeptools deep sequencing data analysis and visualization tools with Pydantic AI integration."""

    def __init__(self, config: MCPServerConfig | None = None):
        if config is None:
            config = MCPServerConfig(
                server_name="deeptools-server",
                server_type=MCPServerType.CUSTOM,
                container_image="python:3.11-slim",
                environment_variables={"DEEPTOOLS_VERSION": "3.5.1"},
                capabilities=[
                    "chip_seq",
                    "rna_seq",
                    "visualization",
                    "data_analysis",
                    "sequencing",
                ],
            )
        super().__init__(config)

    @mcp_tool()
    def deeptools_bam_coverage(
        self,
        bam_file: str,
        output_file: str,
        bin_size: int = 50,
        number_of_processors: int = 1,
        normalize_using: str = "RPGC",
        effective_genome_size: int = 2150570000,
        extend_reads: int = 200,
        ignore_duplicates: bool = False,
        min_mapping_quality: int = 10,
        smooth_length: int = 60,
        scale_factors: str | None = None,
        center_reads: bool = False,
        sam_flag_include: int | None = None,
        sam_flag_exclude: int | None = None,
        min_fragment_length: int = 0,
        max_fragment_length: int = 0,
        use_basal_level: bool = False,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Generate a coverage track from a BAM file using deeptools bamCoverage.

        This tool converts BAM files to bigWig format for visualization in genome browsers.
        It's commonly used for ChIP-seq and RNA-seq data analysis.

        Args:
            bam_file: Input BAM file
            output_file: Output bigWig file path
            bin_size: Size of the bins in bases for coverage calculation
            number_of_processors: Number of processors to use
            normalize_using: Normalization method (RPGC, CPM, BPM, RPKM, None)
            effective_genome_size: Effective genome size for RPGC normalization
            extend_reads: Extend reads to this length
            ignore_duplicates: Ignore duplicate reads
            min_mapping_quality: Minimum mapping quality score
            smooth_length: Smoothing window length
            scale_factors: Scale factors for normalization (file:scale_factor pairs)
            center_reads: Center reads on fragment center
            sam_flag_include: SAM flags to include
            sam_flag_exclude: SAM flags to exclude
            min_fragment_length: Minimum fragment length
            max_fragment_length: Maximum fragment length
            use_basal_level: Use basal level for scaling
            offset: Offset for read positioning

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
        if not os.path.exists(bam_file):
            raise FileNotFoundError(f"Input BAM file not found: {bam_file}")

        # Validate output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "bamCoverage",
                "--bam",
                bam_file,
                "--outFileName",
                output_file,
                "--binSize",
                str(bin_size),
                "--numberOfProcessors",
                str(number_of_processors),
                "--normalizeUsing",
                normalize_using,
            ]

            # Add optional parameters
            if normalize_using == "RPGC":
                cmd.extend(["--effectiveGenomeSize", str(effective_genome_size)])

            if extend_reads > 0:
                cmd.extend(["--extendReads", str(extend_reads)])

            if ignore_duplicates:
                cmd.append("--ignoreDuplicates")

            if min_mapping_quality > 0:
                cmd.extend(["--minMappingQuality", str(min_mapping_quality)])

            if smooth_length > 0:
                cmd.extend(["--smoothLength", str(smooth_length)])

            if scale_factors:
                cmd.extend(["--scaleFactors", scale_factors])

            if center_reads:
                cmd.append("--centerReads")

            if sam_flag_include is not None:
                cmd.extend(["--samFlagInclude", str(sam_flag_include)])

            if sam_flag_exclude is not None:
                cmd.extend(["--samFlagExclude", str(sam_flag_exclude)])

            if min_fragment_length > 0:
                cmd.extend(["--minFragmentLength", str(min_fragment_length)])

            if max_fragment_length > 0:
                cmd.extend(["--maxFragmentLength", str(max_fragment_length)])

            if use_basal_level:
                cmd.append("--useBasalLevel")

            if offset != 0:
                cmd.extend(["--Offset", str(offset)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 minutes timeout
            )

            output_files = [output_file]

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": exc.stdout if exc.stdout else "",
                "stderr": exc.stderr if exc.stderr else "",
                "output_files": [],
                "exit_code": exc.returncode,
                "success": False,
                "error": f"bamCoverage execution failed: {exc}",
            }

        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "bamCoverage timed out after 30 minutes",
            }

    @mcp_tool()
    def deeptools_compute_matrix(
        self,
        regions_file: str,
        score_files: list[str],
        output_file: str,
        reference_point: str = "TSS",
        before_region_start_length: int = 3000,
        after_region_start_length: int = 3000,
        region_body_length: int = 5000,
        bin_size: int = 10,
        missing_data_as_zero: bool = False,
        skip_zeros: bool = False,
        min_mapping_quality: int = 0,
        ignore_duplicates: bool = False,
        scale_factors: str | None = None,
        number_of_processors: int = 1,
        transcript_id_designator: str = "transcript",
        exon_id_designator: str = "exon",
        transcript_id_column: int = 1,
        exon_id_column: int = 1,
        metagene: bool = False,
        smart_labels: bool = False,
    ) -> dict[str, Any]:
        """
        Compute a matrix of scores over genomic regions using deeptools computeMatrix.

        This tool prepares data for heatmap visualization by computing scores over
        specified genomic regions from multiple bigWig files.

        Args:
            regions_file: BED/GTF file containing regions of interest
            score_files: List of bigWig files containing scores
            output_file: Output matrix file (will also create .tab file)
            reference_point: Reference point for matrix computation (TSS, TES, center)
            before_region_start_length: Distance upstream of reference point
            after_region_start_length: Distance downstream of reference point
            region_body_length: Length of region body for scaling
            bin_size: Size of bins for matrix computation
            missing_data_as_zero: Treat missing data as zero
            skip_zeros: Skip zeros in computation
            min_mapping_quality: Minimum mapping quality (for BAM files)
            ignore_duplicates: Ignore duplicate reads (for BAM files)
            scale_factors: Scale factors for normalization
            number_of_processors: Number of processors to use
            transcript_id_designator: Transcript ID designator for GTF files
            exon_id_designator: Exon ID designator for GTF files
            transcript_id_column: Column containing transcript IDs
            exon_id_column: Column containing exon IDs
            metagene: Compute metagene profile
            smart_labels: Use smart labels for output

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        if not os.path.exists(regions_file):
            raise FileNotFoundError(f"Regions file not found: {regions_file}")

        for score_file in score_files:
            if not os.path.exists(score_file):
                raise FileNotFoundError(f"Score file not found: {score_file}")

        # Validate output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "computeMatrix",
                reference_point,
                "--regionsFileName",
                regions_file,
                "--scoreFileName",
                " ".join(score_files),
                "--outFileName",
                output_file,
                "--beforeRegionStartLength",
                str(before_region_start_length),
                "--afterRegionStartLength",
                str(after_region_start_length),
                "--binSize",
                str(bin_size),
                "--numberOfProcessors",
                str(number_of_processors),
            ]

            # Add optional parameters
            if region_body_length > 0:
                cmd.extend(["--regionBodyLength", str(region_body_length)])

            if missing_data_as_zero:
                cmd.append("--missingDataAsZero")

            if skip_zeros:
                cmd.append("--skipZeros")

            if min_mapping_quality > 0:
                cmd.extend(["--minMappingQuality", str(min_mapping_quality)])

            if ignore_duplicates:
                cmd.append("--ignoreDuplicates")

            if scale_factors:
                cmd.extend(["--scaleFactors", scale_factors])

            if transcript_id_designator != "transcript":
                cmd.extend(["--transcriptID", transcript_id_designator])

            if exon_id_designator != "exon":
                cmd.extend(["--exonID", exon_id_designator])

            if transcript_id_column != 1:
                cmd.extend(["--transcript_id_designator", str(transcript_id_column)])

            if exon_id_column != 1:
                cmd.extend(["--exon_id_designator", str(exon_id_column)])

            if metagene:
                cmd.append("--metagene")

            if smart_labels:
                cmd.append("--smartLabels")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )

            output_files = [output_file, f"{output_file}.tab"]

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": exc.stdout if exc.stdout else "",
                "stderr": exc.stderr if exc.stderr else "",
                "output_files": [],
                "exit_code": exc.returncode,
                "success": False,
                "error": f"computeMatrix execution failed: {exc}",
            }

        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "computeMatrix timed out after 1 hour",
            }

    @mcp_tool()
    def deeptools_plot_heatmap(
        self,
        matrix_file: str,
        output_file: str,
        color_map: str = "RdYlBu_r",
        what_to_show: str = "plot, heatmap and colorbar",
        plot_title: str = "",
        x_axis_label: str = "",
        y_axis_label: str = "",
        regions_label: str = "",
        samples_label: str = "",
        legend_location: str = "best",
        plot_width: int = 7,
        plot_height: int = 6,
        dpi: int = 300,
        kmeans: int | None = None,
        hclust: int | None = None,
        sort_regions: str = "no",
        sort_using: str = "mean",
        average_type_summary_plot: str = "mean",
        missing_data_color: str = "black",
        alpha: float = 1.0,
        color_list: str | None = None,
        color_number: int = 256,
        z_min: float | None = None,
        z_max: float | None = None,
        heatmap_height: float = 0.3,
        heatmap_width: float = 0.15,
        what_to_show_colorbar: str = "yes",
    ) -> dict[str, Any]:
        """
        Generate a heatmap from a deeptools matrix using plotHeatmap.

        This tool creates publication-quality heatmaps from deeptools computeMatrix output.

        Args:
            matrix_file: Input matrix file from computeMatrix
            output_file: Output heatmap file (PDF/PNG/SVG)
            color_map: Color map for heatmap
            what_to_show: What to show in the plot
            plot_title: Title for the plot
            x_axis_label: X-axis label
            y_axis_label: Y-axis label
            regions_label: Regions label
            samples_label: Samples label
            legend_location: Location of legend
            plot_width: Width of plot in inches
            plot_height: Height of plot in inches
            dpi: DPI for raster outputs
            kmeans: Number of clusters for k-means clustering
            hclust: Number of clusters for hierarchical clustering
            sort_regions: How to sort regions
            sort_using: What to use for sorting
            average_type_summary_plot: Type of averaging for summary plot
            missing_data_color: Color for missing data
            alpha: Transparency level
            color_list: Custom color list
            color_number: Number of colors in colormap
            z_min: Minimum value for colormap
            z_max: Maximum value for colormap
            heatmap_height: Height of heatmap relative to plot
            heatmap_width: Width of heatmap relative to plot
            what_to_show_colorbar: Whether to show colorbar

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input file exists
        if not os.path.exists(matrix_file):
            raise FileNotFoundError(f"Matrix file not found: {matrix_file}")

        # Validate output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "plotHeatmap",
                "--matrixFile",
                matrix_file,
                "--outFileName",
                output_file,
                "--colorMap",
                color_map,
                "--whatToShow",
                what_to_show,
                "--plotWidth",
                str(plot_width),
                "--plotHeight",
                str(plot_height),
                "--dpi",
                str(dpi),
                "--missingDataColor",
                missing_data_color,
                "--alpha",
                str(alpha),
                "--colorNumber",
                str(color_number),
                "--heatmapHeight",
                str(heatmap_height),
                "--heatmapWidth",
                str(heatmap_width),
                "--whatToShowColorbar",
                what_to_show_colorbar,
            ]

            # Add optional string parameters
            if plot_title:
                cmd.extend(["--plotTitle", plot_title])

            if x_axis_label:
                cmd.extend(["--xAxisLabel", x_axis_label])

            if y_axis_label:
                cmd.extend(["--yAxisLabel", y_axis_label])

            if regions_label:
                cmd.extend(["--regionsLabel", regions_label])

            if samples_label:
                cmd.extend(["--samplesLabel", samples_label])

            if legend_location != "best":
                cmd.extend(["--legendLocation", legend_location])

            if sort_regions != "no":
                cmd.extend(["--sortRegions", sort_regions])

            if sort_using != "mean":
                cmd.extend(["--sortUsing", sort_using])

            if average_type_summary_plot != "mean":
                cmd.extend(["--averageTypeSummaryPlot", average_type_summary_plot])

            # Add optional numeric parameters
            if kmeans is not None and kmeans > 0:
                cmd.extend(["--kmeans", str(kmeans)])

            if hclust is not None and hclust > 0:
                cmd.extend(["--hclust", str(hclust)])

            if color_list:
                cmd.extend(["--colorList", color_list])

            if z_min is not None:
                cmd.extend(["--zMin", str(z_min)])

            if z_max is not None:
                cmd.extend(["--zMax", str(z_max)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 minutes timeout
            )

            output_files = [output_file]

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": exc.stdout if exc.stdout else "",
                "stderr": exc.stderr if exc.stderr else "",
                "output_files": [],
                "exit_code": exc.returncode,
                "success": False,
                "error": f"plotHeatmap execution failed: {exc}",
            }

        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "plotHeatmap timed out after 30 minutes",
            }

    @mcp_tool()
    def deeptools_multi_bam_summary(
        self,
        bam_files: list[str],
        output_file: str,
        bin_size: int = 10000,
        distance_between_bins: int = 0,
        region: str | None = None,
        bed_file: str | None = None,
        labels: list[str] | None = None,
        scaling_factors: str | None = None,
        pcorr: bool = False,
        out_raw_counts: str | None = None,
        extend_reads: int | None = None,
        ignore_duplicates: bool = False,
        min_mapping_quality: int = 0,
        center_reads: bool = False,
        sam_flag_include: int | None = None,
        sam_flag_exclude: int | None = None,
        min_fragment_length: int = 0,
        max_fragment_length: int = 0,
        number_of_processors: int = 1,
    ) -> dict[str, Any]:
        """
        Generate a summary of multiple BAM files using deeptools multiBamSummary.

        This tool computes the read coverage correlation between multiple BAM files,
        useful for comparing ChIP-seq replicates or different conditions.

        Args:
            bam_files: List of input BAM files
            output_file: Output file for correlation matrix
            bin_size: Size of the bins in bases
            distance_between_bins: Distance between bins
            region: Region to analyze (chrom:start-end)
            bed_file: BED file with regions to analyze
            labels: Labels for each BAM file
            scaling_factors: Scaling factors for normalization
            pcorr: Use Pearson correlation instead of Spearman
            out_raw_counts: Output file for raw counts
            extend_reads: Extend reads to this length
            ignore_duplicates: Ignore duplicate reads
            min_mapping_quality: Minimum mapping quality
            center_reads: Center reads on fragment center
            sam_flag_include: SAM flags to include
            sam_flag_exclude: SAM flags to exclude
            min_fragment_length: Minimum fragment length
            max_fragment_length: Maximum fragment length
            number_of_processors: Number of processors to use

        Returns:
            Dictionary containing command executed, stdout, stderr, output files, and exit code
        """
        # Validate input files exist
        for bam_file in bam_files:
            if not os.path.exists(bam_file):
                raise FileNotFoundError(f"BAM file not found: {bam_file}")

        if bed_file and not os.path.exists(bed_file):
            raise FileNotFoundError(f"BED file not found: {bed_file}")

        # Validate output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cmd = [
                "multiBamSummary",
                "bins",
                "--bamfiles",
                " ".join(bam_files),
                "--outFileName",
                output_file,
                "--binSize",
                str(bin_size),
                "--numberOfProcessors",
                str(number_of_processors),
            ]

            # Add optional parameters
            if distance_between_bins > 0:
                cmd.extend(["--distanceBetweenBins", str(distance_between_bins)])

            if region:
                cmd.extend(["--region", region])

            if bed_file:
                cmd.extend(["--BED", bed_file])

            if labels:
                cmd.extend(["--labels", " ".join(labels)])

            if scaling_factors:
                cmd.extend(["--scalingFactors", scaling_factors])

            if pcorr:
                cmd.append("--pcorr")

            if out_raw_counts:
                cmd.extend(["--outRawCounts", out_raw_counts])

            if extend_reads is not None and extend_reads > 0:
                cmd.extend(["--extendReads", str(extend_reads)])

            if ignore_duplicates:
                cmd.append("--ignoreDuplicates")

            if min_mapping_quality > 0:
                cmd.extend(["--minMappingQuality", str(min_mapping_quality)])

            if center_reads:
                cmd.append("--centerReads")

            if sam_flag_include is not None:
                cmd.extend(["--samFlagInclude", str(sam_flag_include)])

            if sam_flag_exclude is not None:
                cmd.extend(["--samFlagExclude", str(sam_flag_exclude)])

            if min_fragment_length > 0:
                cmd.extend(["--minFragmentLength", str(min_fragment_length)])

            if max_fragment_length > 0:
                cmd.extend(["--maxFragmentLength", str(max_fragment_length)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour timeout
            )

            output_files = [output_file]
            if out_raw_counts:
                output_files.append(out_raw_counts)

            return {
                "command_executed": " ".join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": output_files,
                "exit_code": result.returncode,
                "success": True,
                "error": None,
            }

        except subprocess.CalledProcessError as exc:
            return {
                "command_executed": " ".join(cmd),
                "stdout": exc.stdout if exc.stdout else "",
                "stderr": exc.stderr if exc.stderr else "",
                "output_files": [],
                "exit_code": exc.returncode,
                "success": False,
                "error": f"multiBamSummary execution failed: {exc}",
            }

        except subprocess.TimeoutExpired:
            return {
                "command_executed": " ".join(cmd),
                "stdout": "",
                "stderr": "",
                "output_files": [],
                "exit_code": -1,
                "success": False,
                "error": "multiBamSummary timed out after 1 hour",
            }
