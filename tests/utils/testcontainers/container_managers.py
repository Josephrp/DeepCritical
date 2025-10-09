"""
Container management utilities for testing.
"""

from typing import Any, Dict, List, Optional

from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network


class ContainerManager:
    """Manages multiple containers for complex test scenarios."""

    def __init__(self):
        self.containers: dict[str, DockerContainer] = {}
        self.networks: dict[str, Network] = {}

    def add_container(self, name: str, container: DockerContainer) -> None:
        """Add a container to the manager."""
        self.containers[name] = container

    def add_network(self, name: str, network: Network) -> None:
        """Add a network to the manager."""
        self.networks[name] = network

    def start_all(self) -> None:
        """Start all managed containers."""
        for container in self.containers.values():
            container.start()

    def stop_all(self) -> None:
        """Stop all managed containers."""
        for container in self.containers.values():
            try:
                container.stop()
            except Exception:
                pass  # Ignore errors during cleanup

    def get_container(self, name: str) -> DockerContainer | None:
        """Get a container by name."""
        return self.containers.get(name)

    def get_network(self, name: str) -> Network | None:
        """Get a network by name."""
        return self.networks.get(name)

    def cleanup(self) -> None:
        """Clean up all containers and networks."""
        self.stop_all()

        for network in self.networks.values():
            try:
                network.remove()
            except Exception:
                pass  # Ignore errors during cleanup


class VLLMContainer(DockerContainer):
    """Specialized container for VLLM testing."""

    def __init__(self, model: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__("vllm/vllm-openai:latest", **kwargs)
        self.model = model
        self._configure_vllm()

    def _configure_vllm(self) -> None:
        """Configure VLLM-specific settings."""
        self.with_env("VLLM_MODEL", self.model)
        self.with_env("VLLM_HOST", "0.0.0.0")
        self.with_env("VLLM_PORT", "8000")

    def get_connection_url(self) -> str:
        """Get the connection URL for the VLLM server."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port("8000")
        return f"http://{host}:{port}"


class BioinformaticsContainer(DockerContainer):
    """Specialized container for bioinformatics tools testing."""

    def __init__(self, tool: str = "bwa", **kwargs):
        super().__init__(f"biocontainers/{tool}:latest", **kwargs)
        self.tool = tool

    def get_tool_version(self) -> str:
        """Get the version of the bioinformatics tool."""
        result = self.exec(f"{self.tool} --version")
        return result.output.decode().strip()

    def get_connection_url(self) -> str:
        """Get the connection URL for the container."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port("8000")
        return f"http://{host}:{port}"
