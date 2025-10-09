"""
LLaMACPP containerized model loading tests.
"""

import time

import pytest
import requests
from testcontainers.core.container import DockerContainer


class TestLLaMACPPModelLoading:
    """Test LLaMACPP model loading in containerized environment."""

    @pytest.mark.containerized
    def test_llamacpp_model_loading_success(self):
        """Test successful LLaMACPP model loading in container."""
        # Skip this test since LLaMACPP containers aren't available in the testcontainers fork
        pytest.skip(
            "LLaMACPP container testing not available in current testcontainers version"
        )

        # Create container for testing

        import uuid

        # Create unique container name with timestamp to avoid conflicts
        container_name = (
            f"test-bioinformatics-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        )
        container = DockerContainer("python:3.11-slim")
        container.with_name(container_name)
        container.with_exposed_ports("8003")

        with container:
            container.start()

            # Wait for model to load
            max_wait = 300  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < max_wait:
                try:
                    # Get connection URL manually since basic DockerContainer doesn't have get_connection_url
                    host = container.get_container_host_ip()
                    port = container.get_exposed_port(8003)
                    response = requests.get(f"http://{host}:{port}/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    time.sleep(5)
            else:
                pytest.fail("LLaMACPP model failed to load within timeout")

            # Verify model metadata
            # Get connection URL manually
            host = container.get_container_host_ip()
            port = container.get_exposed_port(8003)
            info_response = requests.get(f"http://{host}:{port}/v1/models")
            models = info_response.json()
            assert len(models["data"]) > 0
            assert "DialoGPT" in models["data"][0]["id"]

    @pytest.mark.containerized
    def test_llamacpp_text_generation(self):
        """Test text generation with LLaMACPP."""
        # Skip this test since LLaMACPP containers aren't available in the testcontainers fork
        pytest.skip(
            "LLaMACPP container testing not available in current testcontainers version"
        )

        # Create container for testing

        import uuid

        # Create unique container name with timestamp to avoid conflicts
        container_name = (
            f"test-bioinformatics-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        )
        container = DockerContainer("python:3.11-slim")
        container.with_name(container_name)
        container.with_exposed_ports("8003")

        with container:
            container.start()

            # Wait for model to be ready
            time.sleep(60)

            # Test text generation
            payload = {
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.7,
            }

            # Get connection URL manually
            host = container.get_container_host_ip()
            port = container.get_exposed_port(8003)
            response = requests.post(
                f"http://{host}:{port}/v1/completions", json=payload
            )

            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "text" in result["choices"][0]

    @pytest.mark.containerized
    def test_llamacpp_error_handling(self):
        """Test error handling for invalid requests."""
        # Skip this test since LLaMACPP containers aren't available in the testcontainers fork
        pytest.skip(
            "LLaMACPP container testing not available in current testcontainers version"
        )
