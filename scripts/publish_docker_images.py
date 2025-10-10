#!/usr/bin/env python3
"""
Script to build and publish bioinformatics Docker images to Docker Hub.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

# Docker Hub configuration - uses environment variables with defaults
DOCKER_HUB_USERNAME = os.getenv(
    "DOCKER_HUB_USERNAME", "tonic01"
)  # Replace with your Docker Hub username
DOCKER_HUB_REPO = os.getenv("DOCKER_HUB_REPO", "deepcritical-bioinformatics")
TAG = os.getenv("DOCKER_TAG", "latest")

# List of bioinformatics tools to build
BIOINFORMATICS_TOOLS = [
    "bcftools",
    "bedtools",
    "bowtie2",
    "busco",
    "bwa",
    "cutadapt",
    "deeptools",
    "fastp",
    "fastqc",
    "featurecounts",
    "flye",
    "freebayes",
    "hisat2",
    "homer",
    "htseq",
    "kallisto",
    "macs3",
    "meme",
    "minimap2",
    "multiqc",
    "picard",
    "qualimap",
    "salmon",
    "samtools",
    "seqtk",
    "star",
    "stringtie",
    "tophat",
    "trimgalore",
]


def check_image_exists(tool_name: str) -> bool:
    """Check if a Docker Hub image exists."""
    image_name = f"{DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}-{tool_name}:{TAG}"
    try:
        # Try to pull the image manifest to check if it exists
        result = subprocess.run(
            ["docker", "manifest", "inspect", image_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


async def build_and_publish_image(tool_name: str):
    """Build and publish a single Docker image."""
    print(f"\n{'=' * 50}")
    print(f"Building and publishing {tool_name}")
    print(f"{'=' * 50}")

    dockerfile_path = f"docker/bioinformatics/Dockerfile.{tool_name}"
    image_name = f"{DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}-{tool_name}:{TAG}"

    try:
        # Build the image
        print(f"Building Docker image: {image_name}")
        build_cmd = ["docker", "build", "-f", dockerfile_path, "-t", image_name, "."]

        subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] Successfully built {image_name}")

        # Tag as latest
        tag_cmd = [
            "docker",
            "tag",
            image_name,
            f"{DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}-{tool_name}:latest",
        ]
        subprocess.run(tag_cmd, check=True)
        print("[SUCCESS] Tagged as latest")

        # Push to Docker Hub
        print(f"Pushing to Docker Hub: {image_name}")
        push_cmd = ["docker", "push", image_name]
        subprocess.run(push_cmd, check=True)
        print(f"[SUCCESS] Successfully pushed {image_name}")

        # Push latest tag
        latest_image = f"{DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}-{tool_name}:latest"
        push_latest_cmd = ["docker", "push", latest_image]
        subprocess.run(push_latest_cmd, check=True)
        print(f"[SUCCESS] Successfully pushed {latest_image}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to build/publish {tool_name}: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error for {tool_name}: {e}")
        return False


async def check_images_only():
    """Check which Docker Hub images exist without building."""
    print("🔍 Checking Docker Hub image availability...")
    print(f"Repository: {DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}")
    print(f"Tag: {TAG}")
    print()

    available_images = []
    missing_images = []

    for tool in BIOINFORMATICS_TOOLS:
        if check_image_exists(tool):
            print(f"✅ {tool}: Available")
            available_images.append(tool)
        else:
            print(f"❌ {tool}: Not found")
            missing_images.append(tool)

    print(f"\n{'=' * 50}")
    print("📊 Image Availability Summary:")
    print(f"✅ Available: {len(available_images)}")
    print(f"❌ Missing: {len(missing_images)}")
    print(
        f"📈 Availability: {(len(available_images) / len(BIOINFORMATICS_TOOLS)) * 100:.1f}%"
    )
    print(f"{'=' * 50}")

    if missing_images:
        print("\n📝 Missing images:")
        for tool in missing_images:
            print(f"  - {DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}-{tool}:{TAG}")


async def main():
    """Main function to build and publish all images."""
    parser = argparse.ArgumentParser(
        description="Build and publish bioinformatics Docker images"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which images exist on Docker Hub",
    )
    args = parser.parse_args()

    if args.check_only:
        await check_images_only()
        return

    print("[START] Starting Docker Hub publishing process...")
    print(f"Repository: {DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}")
    print(f"Tools to process: {len(BIOINFORMATICS_TOOLS)}")

    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("[OK] Docker is available")
    except subprocess.CalledProcessError:
        print("[ERROR] Docker is not available. Please install Docker first.")
        return

    # Check if Docker daemon is running
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        print("[OK] Docker daemon is running")
    except subprocess.CalledProcessError:
        print("[ERROR] Docker daemon is not running. Please start Docker first.")
        return

    successful_builds = 0
    failed_builds = 0

    # Build and publish each image
    for tool in BIOINFORMATICS_TOOLS:
        success = await build_and_publish_image(tool)
        if success:
            successful_builds += 1
        else:
            failed_builds += 1

    print(f"\n{'=' * 50}")
    print("[SUMMARY] Publishing Summary:")
    print(f"[SUCCESS] Successful builds: {successful_builds}")
    print(f"[FAILED] Failed builds: {failed_builds}")
    print(
        f"[RATE] Success rate: {(successful_builds / len(BIOINFORMATICS_TOOLS)) * 100:.1f}%"
    )
    print(f"{'=' * 50}")

    if failed_builds > 0:
        print("\n[WARNING] Some builds failed. Check the output above for details.")
        print("You may need to:")
        print("- Check Docker Hub credentials")
        print("- Verify Dockerfile syntax")
        print("- Ensure all dependencies are available")
        print("- Check available disk space")
    else:
        print("\n[SUCCESS] All images successfully built and published!")
        print("\n[USAGE] Usage:")
        print("Update your bioinformatics server configs to use:")
        print(
            f'container_image = "{DOCKER_HUB_USERNAME}/{DOCKER_HUB_REPO}-{{tool_name}}:{TAG}"'
        )


if __name__ == "__main__":
    asyncio.run(main())
