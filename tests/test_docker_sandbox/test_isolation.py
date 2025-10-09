"""
Docker sandbox isolation tests for security validation.
"""

import os
import subprocess
from pathlib import Path

import pytest

from DeepResearch.src.tools.docker_sandbox import DockerSandboxRunner
from tests.utils.testcontainers.docker_helpers import create_isolated_container


class TestDockerSandboxIsolation:
    """Test container isolation and security."""

    @pytest.mark.containerized
    def test_container_cannot_access_proc(self, test_config):
        """Test that container cannot access /proc filesystem."""
        if not test_config["docker_enabled"]:
            pytest.skip("Docker tests disabled")

        # Create container with restricted access
        container = create_isolated_container(
            image="python:3.11-slim",
            command=["python", "-c", "import os; print(open('/proc/version').read())"],
        )

        with container:
            container.start()
            result = container.get_wrapped_container().exec_run(
                ["python", "-c", "import os; print(open('/proc/version').read())"]
            )

            # Should fail with permission denied
            assert result.exit_code != 0
            assert (
                b"Permission denied" in result.output[1] if result.output[1] else True
            )

    @pytest.mark.containerized
    def test_container_cannot_access_host_dirs(self, test_config):
        """Test that container cannot access unauthorized host directories."""
        if not test_config["docker_enabled"]:
            pytest.skip("Docker tests disabled")

        container = create_isolated_container(
            image="python:3.11-slim",
            command=["python", "-c", "import os; print(open('/etc/passwd').read())"],
        )

        with container:
            container.start()
            result = container.get_wrapped_container().exec_run(
                ["python", "-c", "import os; print(open('/etc/passwd').read())"]
            )

            # Should fail with permission denied
            assert result.exit_code != 0

    @pytest.mark.containerized
    def test_readonly_mounts_enforced(self, test_config, tmp_path):
        """Test that read-only mounts cannot be written to."""
        if not test_config["docker_enabled"]:
            pytest.skip("Docker tests disabled")

        # Create test file
        test_file = tmp_path / "readonly_test.txt"
        test_file.write_text("test content")

        container = create_isolated_container(
            image="python:3.11-slim",
            volumes={str(test_file): {"bind": "/test/readonly.txt", "mode": "ro"}},
            command=[
                "python",
                "-c",
                "open('/test/readonly.txt', 'w').write('modified')",
            ],
        )

        with container:
            container.start()
            result = container.get_wrapped_container().exec_run(
                ["python", "-c", "open('/test/readonly.txt', 'w').write('modified')"]
            )

            # Should fail with permission denied
            assert result.exit_code != 0

            # Verify original content unchanged
            assert test_file.read_text() == "test content"
