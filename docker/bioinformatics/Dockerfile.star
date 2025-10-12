# STAR Docker container for RNA-seq alignment
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install STAR via conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda config --set auto_update_conda false && \
    /opt/conda/bin/conda config --set safety_checks disabled && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda config --add channels bioconda && \
    /opt/conda/bin/conda config --add channels conda-forge && \
    /opt/conda/bin/conda install -c bioconda -c conda-forge star -y && \
    ln -s /opt/conda/bin/STAR /usr/local/bin/STAR

# Create working directory
WORKDIR /workspace

# Set environment variables
ENV STAR_VERSION=2.7.10b

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD STAR --version || exit 1

# Default command
CMD ["STAR", "--help"]
