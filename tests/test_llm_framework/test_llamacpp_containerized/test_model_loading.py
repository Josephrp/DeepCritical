"""
LLaMACPP containerized model loading tests.
"""

import time

import pytest
import requests

from tests.utils.testcontainers.container_managers import BioinformaticsContainer


class TestLLaMACPPModelLoading:
    """Test LLaMACPP model loading in containerized environment."""

    @pytest.mark.containerized
    def test_llamacpp_model_loading_success(self):
        """Test successful LLaMACPP model loading in container."""
        # Note: LLaMACPP container testing would require a different container setup
        # For now, we'll test with a bioinformatics container as a placeholder
        container = BioinformaticsContainer(tool="bwa", ports={"8001": "8001"})

        with container:
            container.start()

            # Wait for model to load
            max_wait = 300  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < max_wait:
                try:
                    response = requests.get(f"{container.get_connection_url()}/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    time.sleep(5)
            else:
                pytest.fail("LLaMACPP model failed to load within timeout")

            # Verify model metadata
            info_response = requests.get(f"{container.get_connection_url()}/v1/models")
            models = info_response.json()
            assert len(models["data"]) > 0
            assert "DialoGPT" in models["data"][0]["id"]

    @pytest.mark.containerized
    def test_llamacpp_text_generation(self):
        """Test text generation with LLaMACPP."""
        # Note: LLaMACPP container testing would require a different container setup
        # For now, we'll test with a bioinformatics container as a placeholder
        container = BioinformaticsContainer(tool="bwa", ports={"8002": "8002"})

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

            response = requests.post(
                f"{container.get_connection_url()}/v1/completions", json=payload
            )

            assert response.status_code == 200
            result = response.json()
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "text" in result["choices"][0]

    @pytest.mark.containerized
    def test_llamacpp_error_handling(self):
        """Test error handling for invalid requests."""
        # Note: LLaMACPP container testing would require a different container setup
        # For now, we'll test with a bioinformatics container as a placeholder
        container = BioinformaticsContainer(tool="bwa", ports={"8003": "8003"})

        with container:
            container.start()
            time.sleep(30)

            # Test invalid request
            payload = {
                "prompt": "",  # Empty prompt should fail
                "max_tokens": 0,
            }

            response = requests.post(
                f"{container.get_connection_url()}/v1/completions", json=payload
            )

            # Should return error status
            assert response.status_code != 200
