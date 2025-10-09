"""
Docker container fixtures for testing.
"""

import pytest

from tests.utils.testcontainers.docker_helpers import create_isolated_container


@pytest.fixture
def isolated_python_container():
    """Fixture for isolated Python container."""
    container = create_isolated_container(
        image="python:3.11-slim", command=["python", "-c", "print('Container ready')"]
    )
    return container


@pytest.fixture
def vllm_container():
    """Fixture for VLLM test container."""
    container = create_isolated_container(
        image="vllm/vllm-openai:latest",
        command=[
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "microsoft/DialoGPT-small",
        ],
        ports={"8000": "8000"},
    )
    return container


@pytest.fixture
def bioinformatics_container():
    """Fixture for bioinformatics tools container."""
    container = create_isolated_container(
        image="biocontainers/bwa:latest", command=["bwa", "--version"]
    )
    return container
