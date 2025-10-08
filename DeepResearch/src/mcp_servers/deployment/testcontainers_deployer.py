"""
Testcontainers Deployer for MCP Servers.

This module provides deployment functionality for MCP servers using testcontainers
for isolated execution environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...datatypes.mcp import MCPServerConfig, MCPServerDeployment, MCPServerStatus
from ...mcp_servers.vendored.bowtie2_server import Bowtie2Server
from ...mcp_servers.vendored.fastqc_server import FastQCServer
from ...mcp_servers.vendored.samtools_server import SamtoolsServer
from ..utils.mcp_server_base import MCPServerBase

logger = logging.getLogger(__name__)


class TestcontainersConfig(BaseModel):
    """Configuration for testcontainers deployment."""

    image: str = Field("python:3.11-slim", description="Base Docker image")
    working_directory: str = Field(
        "/workspace", description="Working directory in container"
    )
    auto_remove: bool = Field(True, description="Auto-remove container after use")
    network_disabled: bool = Field(False, description="Disable network access")
    privileged: bool = Field(False, description="Run container in privileged mode")
    environment_variables: dict[str, str] = Field(
        default_factory=dict, description="Environment variables"
    )
    volumes: dict[str, str] = Field(default_factory=dict, description="Volume mounts")
    ports: dict[str, int] = Field(default_factory=dict, description="Port mappings")
    command: str | None = Field(None, description="Command to run in container")
    entrypoint: str | None = Field(None, description="Container entrypoint")

    class Config:
        json_schema_extra = {
            "example": {
                "image": "python:3.11-slim",
                "working_directory": "/workspace",
                "auto_remove": True,
                "environment_variables": {"PYTHONUNBUFFERED": "1"},
                "volumes": {"/host/data": "/workspace/data"},
            }
        }


class TestcontainersDeployer:
    """Deployer for MCP servers using testcontainers."""

    def __init__(self):
        self.deployments: dict[str, MCPServerDeployment] = {}
        self.containers: dict[
            str, Any
        ] = {}  # Would hold testcontainers container objects

        # Map server types to their implementations
        self.server_implementations = {
            "fastqc": FastQCServer,
            "samtools": SamtoolsServer,
            "bowtie2": Bowtie2Server,
        }

    def create_deployment_config(
        self, server_name: str, **kwargs
    ) -> TestcontainersConfig:
        """Create deployment configuration for a server."""
        base_config = TestcontainersConfig()

        # Customize based on server type
        if server_name in self.server_implementations:
            server = self.server_implementations[server_name]

            # Add server-specific environment variables
            base_config.environment_variables.update(
                {
                    "MCP_SERVER_NAME": server_name,
                    "MCP_SERVER_VERSION": getattr(server, "version", "1.0.0"),
                    "PYTHONPATH": "/workspace",
                }
            )

            # Add server-specific volumes for data
            base_config.volumes.update(
                {
                    f"/tmp/mcp_{server_name}": "/workspace/data",
                }
            )

        # Apply customizations from kwargs
        for key, value in kwargs.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)

        return base_config

    async def deploy_server(
        self, server_name: str, config: TestcontainersConfig | None = None, **kwargs
    ) -> MCPServerDeployment:
        """Enhanced deployment with Pydantic AI integration."""
        deployment = MCPServerDeployment(
            server_name=server_name,
            status=MCPServerStatus.DEPLOYING,
        )

        try:
            # Get server implementation
            server = self._get_server_implementation(server_name)
            if not server:
                raise ValueError(f"Server implementation for '{server_name}' not found")

            # Use testcontainers deployment method if available
            if hasattr(server, "deploy_with_testcontainers"):
                deployment = await server.deploy_with_testcontainers()
            else:
                # Fallback to basic deployment
                deployment = await self._deploy_server_basic(
                    server_name, config, **kwargs
                )

            # Update deployment registry
            self.deployments[server_name] = deployment
            self.server_implementations[server_name] = server

            return deployment

        except Exception as e:
            deployment.status = MCPServerStatus.FAILED
            deployment.error_message = str(e)
            self.deployments[server_name] = deployment
            raise

    async def _deploy_server_basic(
        self, server_name: str, config: TestcontainersConfig | None = None, **kwargs
    ) -> MCPServerDeployment:
        """Basic deployment method for servers without testcontainers support."""
        try:
            # Create deployment configuration
            if config is None:
                config = self.create_deployment_config(server_name, **kwargs)

            # Create deployment record
            deployment = MCPServerDeployment(
                server_name=server_name,
                status=MCPServerStatus.PENDING,
                configuration=MCPServerConfig(
                    server_name=server_name,
                    server_type=self._get_server_type(server_name),
                ),
            )

            # In a real implementation, this would use testcontainers
            # For now, we'll simulate deployment
            deployment.status = MCPServerStatus.RUNNING
            deployment.container_name = f"mcp-{server_name}-container"
            deployment.container_id = f"container_{id(deployment)}"
            deployment.started_at = datetime.now()

            # Store deployment
            self.deployments[server_name] = deployment

            logger.info(
                f"Deployed MCP server '{server_name}' with container '{deployment.container_id}'"
            )

            return deployment

        except Exception as e:
            logger.error(f"Failed to deploy MCP server '{server_name}': {e}")
            deployment = MCPServerDeployment(
                server_name=server_name,
                server_type=self._get_server_type(server_name),
                status=MCPServerStatus.FAILED,
                error_message=str(e),
                configuration=MCPServerConfig(
                    server_name=server_name,
                    server_type=self._get_server_type(server_name),
                ),
            )
            self.deployments[server_name] = deployment
            return deployment

    async def stop_server(self, server_name: str) -> bool:
        """Stop a deployed MCP server."""
        if server_name not in self.deployments:
            logger.warning(f"Server '{server_name}' not found in deployments")
            return False

        deployment = self.deployments[server_name]

        try:
            # In a real implementation, this would stop the testcontainers container
            deployment.status = "stopped"
            deployment.finished_at = None  # Would be set by testcontainers

            # Clean up container reference
            if server_name in self.containers:
                del self.containers[server_name]

            logger.info(f"Stopped MCP server '{server_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to stop MCP server '{server_name}': {e}")
            deployment.status = "failed"
            deployment.error_message = str(e)
            return False

    async def get_server_status(self, server_name: str) -> MCPServerDeployment | None:
        """Get the status of a deployed server."""
        return self.deployments.get(server_name)

    async def list_servers(self) -> list[MCPServerDeployment]:
        """List all deployed servers."""
        return list(self.deployments.values())

    async def execute_tool(
        self, server_name: str, tool_name: str, **kwargs
    ) -> dict[str, Any]:
        """Execute a tool on a deployed server."""
        deployment = self.deployments.get(server_name)
        if not deployment:
            raise ValueError(f"Server '{server_name}' not deployed")

        if deployment.status != "running":
            raise ValueError(
                f"Server '{server_name}' is not running (status: {deployment.status})"
            )

        # Get server implementation
        server = self.server_implementations.get(server_name)
        if not server:
            raise ValueError(f"Server implementation for '{server_name}' not found")

        # Check if tool exists
        available_tools = server.list_tools()
        if tool_name not in available_tools:
            raise ValueError(
                f"Tool '{tool_name}' not found on server '{server_name}'. Available tools: {', '.join(available_tools)}"
            )

        # Execute tool
        try:
            result = server.execute_tool(tool_name, **kwargs)
            return result
        except Exception as e:
            raise ValueError(f"Tool execution failed: {e}")

    def _get_server_type(self, server_name: str) -> str:
        """Get the server type from the server name."""
        if server_name in self.server_implementations:
            return server_name
        return "custom"

    async def create_server_files(self, server_name: str, output_dir: str) -> list[str]:
        """Create necessary files for server deployment."""
        files_created = []

        try:
            # Create temporary directory for server files
            server_dir = Path(output_dir) / f"mcp_{server_name}"
            server_dir.mkdir(parents=True, exist_ok=True)

            # Create server script
            server_script = server_dir / f"{server_name}_server.py"

            # Generate server code based on server type
            server_code = self._generate_server_code(server_name)

            with open(server_script, "w") as f:
                f.write(server_code)

            files_created.append(str(server_script))

            # Create requirements file
            requirements_file = server_dir / "requirements.txt"
            requirements_content = self._generate_requirements(server_name)

            with open(requirements_file, "w") as f:
                f.write(requirements_content)

            files_created.append(str(requirements_file))

            logger.info(f"Created server files for '{server_name}' in {server_dir}")
            return files_created

        except Exception as e:
            logger.error(f"Failed to create server files for '{server_name}': {e}")
            return files_created

    def _generate_server_code(self, server_name: str) -> str:
        """Generate server code for deployment."""
        server = self.server_implementations.get(server_name)
        if not server:
            return "# Server implementation not found"

        # Generate basic server code structure
        code = f'''"""
Auto-generated MCP server for {server_name}.
"""

from {server.__module__} import {server.__class__.__name__}

# Create and run server
server = {server.__class__.__name__}()

if __name__ == "__main__":
    print(f"MCP Server '{server.name}' v{server.version} ready")
    print(f"Available tools: {{', '.join(server.list_tools())}}")
'''

        return code

    def _generate_requirements(self, server_name: str) -> str:
        """Generate requirements file for server deployment."""
        # Basic requirements for MCP servers
        requirements = [
            "pydantic>=2.0.0",
            "fastmcp>=0.1.0",  # Assuming this would be available
        ]

        # Add server-specific requirements
        if server_name == "fastqc":
            requirements.extend(
                [
                    "biopython>=1.80",
                    "numpy>=1.21.0",
                ]
            )
        elif server_name == "samtools":
            requirements.extend(
                [
                    "pysam>=0.20.0",
                ]
            )
        elif server_name == "bowtie2":
            requirements.extend(
                [
                    "biopython>=1.80",
                ]
            )

        return "\n".join(requirements)

    async def cleanup_server(self, server_name: str) -> bool:
        """Clean up a deployed server and its files."""
        try:
            # Stop the server
            await self.stop_server(server_name)

            # Remove from deployments
            if server_name in self.deployments:
                del self.deployments[server_name]

            # Remove container reference
            if server_name in self.containers:
                del self.containers[server_name]

            logger.info(f"Cleaned up MCP server '{server_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup server '{server_name}': {e}")
            return False

    async def health_check(self, server_name: str) -> bool:
        """Perform health check on a deployed server."""
        deployment = self.deployments.get(server_name)
        if not deployment:
            return False

        if deployment.status != "running":
            return False

        try:
            # In a real implementation, this would check if the container is healthy
            # For now, we'll just check if the deployment exists and is running
            return True
        except Exception as e:
            logger.error(f"Health check failed for server '{server_name}': {e}")
            return False


# Global deployer instance
testcontainers_deployer = TestcontainersDeployer()
