# Bioinformatics Tools Docker Containers

This directory contains Dockerfiles for all bioinformatics tools used in the DeepCritical project. Each Dockerfile is optimized for the specific tool and includes all necessary dependencies.

## Available Containers

| Tool | Dockerfile | Description |
|------|------------|-------------|
| **BCFtools** | `Dockerfile.bcftools` | Variant analysis and manipulation |
| **BEDTools** | `Dockerfile.bedtools` | Genomic arithmetic operations |
| **Bowtie2** | `Dockerfile.bowtie2` | Sequence alignment tool |
| **BUSCO** | `Dockerfile.busco` | Genome completeness assessment |
| **BWA** | `Dockerfile.bwa` | DNA sequence alignment |
| **Cutadapt** | `Dockerfile.cutadapt` | Adapter trimming |
| **Deeptools** | `Dockerfile.deeptools` | Deep sequencing data analysis |
| **Fastp** | `Dockerfile.fastp` | FASTQ preprocessing |
| **FastQC** | `Dockerfile.fastqc` | Quality control |
| **featureCounts** | `Dockerfile.featurecounts` | Read counting |
| **Flye** | `Dockerfile.flye` | Long-read genome assembly |
| **FreeBayes** | `Dockerfile.freebayes` | Bayesian variant calling |
| **HISAT2** | `Dockerfile.hisat2` | RNA-seq alignment |
| **HOMER** | `Dockerfile.homer` | Motif analysis |
| **HTSeq** | `Dockerfile.htseq` | Read counting |
| **Kallisto** | `Dockerfile.kallisto` | RNA-seq quantification |
| **MACS3** | `Dockerfile.macs3` | ChIP-seq peak calling |
| **MEME** | `Dockerfile.meme` | Motif discovery |
| **Minimap2** | `Dockerfile.minimap2` | Versatile pairwise alignment |
| **MultiQC** | `Dockerfile.multiqc` | Report generation |
| **Picard** | `Dockerfile.picard` | SAM/BAM processing |
| **Qualimap** | `Dockerfile.qualimap` | Quality control |
| **Salmon** | `Dockerfile.salmon` | RNA-seq quantification |
| **Samtools** | `Dockerfile.samtools` | SAM/BAM processing |
| **Seqtk** | `Dockerfile.seqtk` | FASTA/Q processing |
| **STAR** | `Dockerfile.star` | RNA-seq alignment |
| **StringTie** | `Dockerfile.stringtie` | Transcript assembly |
| **TopHat** | `Dockerfile.tophat` | RNA-seq splice-aware alignment |
| **TrimGalore** | `Dockerfile.trimgalore` | Adapter trimming |

## Usage

### Building Individual Containers

```bash
# Build a specific tool container
docker build -f docker/bioinformatics/Dockerfile.bcftools -t deepcritical-bcftools:latest .

# Build all containers
for dockerfile in docker/bioinformatics/Dockerfile.*; do
    tool=$(basename "$dockerfile" | cut -d'.' -f2)
    docker build -f "$dockerfile" -t "deepcritical-${tool}:latest" .
done
```

### Running Containers

```bash
# Run BCFtools container
docker run --rm -v $(pwd):/data deepcritical-bcftools:latest bcftools view -h /data/sample.vcf

# Run with interactive shell
docker run --rm -it -v $(pwd):/workspace deepcritical-bcftools:latest /bin/bash
```

### Using in Python Applications

```python
from DeepResearch.src.tools.bioinformatics.bcftools_server import BCFtoolsServer

# Create server instance
server = BCFtoolsServer()

# Deploy with Docker
deployment = await server.deploy_with_testcontainers()
print(f"Container ID: {deployment.container_id}")
```

## Configuration

Each Dockerfile includes:

- **Base Image**: Python 3.11-slim for consistency
- **System Dependencies**: All required libraries and tools
- **Python Dependencies**: Tool-specific Python packages
- **Health Checks**: Container health monitoring
- **Environment Variables**: Tool-specific configuration
- **Working Directory**: Consistent `/workspace` setup

## Testing

All containers include health checks and can be tested using:

```bash
# Test container health
docker run --rm deepcritical-bcftools:latest bcftools --version

# Run bioinformatics tests
make test-bioinformatics
```

## Dependencies

### System Level
- **Compression**: zlib, libbz2, liblzma
- **Bioinformatics**: htslib (samtools, bcftools)
- **Java**: OpenJDK 11 (FastQC, Picard, Qualimap)
- **R**: R-base (Qualimap)
- **Perl**: Perl libraries (HOMER, MEME)

### Python Level
- **Core**: numpy, scipy, matplotlib
- **Bioinformatics**: pysam, biopython
- **Specialized**: cutadapt, deeptools, HTSeq, etc.

## Optimization Notes

- All containers use multi-stage builds where applicable
- Health checks prevent running failed containers
- Minimal base images reduce attack surface
- Proper dependency management ensures reproducibility
- Volume mounting support for data persistence

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure Docker daemon is running and user has permissions
2. **Port Conflicts**: Use different ports or check existing containers
3. **Memory Issues**: Increase Docker memory limits for large datasets
4. **Network Issues**: Check firewall settings and network configuration

### Debug Commands

```bash
# Check container logs
docker logs <container_id>

# Inspect container
docker inspect <container_id>

# Enter running container
docker exec -it <container_id> /bin/bash
```

## Contributing

When adding new bioinformatics tools:

1. Create a new Dockerfile following the established pattern
2. Add appropriate tests in `tests/test_bioinformatics_tools/`
3. Update this README with tool information
4. Test the container thoroughly before deployment

## License

These Dockerfiles are part of the DeepCritical project and follow the same licensing terms.

[69 tools called]

## Summary

**Dockerfiles created for:**
- [x] bcftools
- [x] bedtools
- [x] bowtie2
- [x] busco
- [x] bwa
- [x] cutadapt
- [x] deeptools
- [x] fastp
- [x] fastqc
- [x] featurecounts
- [x] flye
- [x] freebayes
- [x] hisat2
- [x] homer
- [x] htseq
- [x] kallisto
- [x] macs3
- [x] meme
- [x] minimap2
- [x] multiqc
- [x] picard
- [x] qualimap
- [x] salmon
- [x] samtools
- [x] seqtk
- [x] star
- [x] stringtie
- [x] tophat
- [x] trimgalore

**Key features of each Dockerfile:**
- Python 3.11-slim base image for consistency
- All required system dependencies pre-installed
- Tool-specific Python packages
- Health checks for container monitoring
- Proper environment variable configuration
- Working directory setup

### ✅ Test Suite Expansion

**test files for:**

- [x] bcftools_server
- [x] bowtie2_server
- [x] busco_server
- [x] cutadapt_server
- [x] deeptools_server
- [x] fastp_server
- [x] fastqc_server
- [x] flye_server
- [x] homer_server
- [x] htseq_server
- [x] kallisto_server
- [x] macs3_server
- [x] meme_server
- [x] minimap2_server
- [x] multiqc_server
- [x] picard_server
- [x] qualimap_server
- [x] salmon_server
- [x] seqtk_server
- [x] stringtie_server
- [x] tophat_server
- [x] trimgalore_server

**Test structure follows existing patterns:**
- Inherits from `BaseBioinformaticsToolTest`
- Includes sample data fixtures
- Tests basic functionality, parameter validation, and error handling
- All marked with `@pytest.mark.optional` for proper test organization


### 🚀 Useage


1. **Build containers:**
   ```bash
   docker build -f docker/bioinformatics/Dockerfile.bcftools -t deepcritical-bcftools:latest .
   ```

2. **Run bioinformatics tests:**
   ```bash
   make test-bioinformatics
   ```

3. **Use in bioinformatics workflows:**
   ```python
   from DeepResearch.src.tools.bioinformatics.bcftools_server import BCFtoolsServer
   server = BCFtoolsServer()
   deployment = await server.deploy_with_testcontainers()
   ```

The implementation provides a complete containerized environment for all bioinformatics tools used in DeepCritical, ensuring reproducibility and easy deployment across different environments.
