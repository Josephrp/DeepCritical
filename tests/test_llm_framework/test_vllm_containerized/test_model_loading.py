"""
VLLM containerized model loading tests.
"""

import time

import pytest
import requests

from tests.utils.testcontainers.container_managers import VLLMContainer


class TestVLLMModelLoading:
    """Test VLLM model loading in containerized environment."""

    @pytest.mark.containerized
    def test_model_loading_success(self):
        """Test successful model loading in container."""
        # Skip VLLM tests for now due to persistent device detection issues in containerized environment
        # pytest.skip("VLLM containerized tests disabled due to device detection issues")

        container = VLLMContainer(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", ports={"8000": "8000"}
        )

        with container:
            container.start()

            # Wait for model to load
            max_wait = 600  # 5 minutes
            start_time = time.time()

            while time.time() - start_time < max_wait:
                try:
                    response = requests.get(f"{container.get_connection_url()}/health")
                    if response.status_code == 200:
                        break
                except Exception:
                    time.sleep(5)
            else:
                pytest.fail("Model failed to load within timeout")

            # Verify model metadata
            info_response = requests.get(f"{container.get_connection_url()}/v1/models")
            models = info_response.json()
            assert len(models["data"]) > 0
            assert "DialoGPT" in models["data"][0]["id"]

    @pytest.mark.containerized
    def test_model_loading_failure(self):
        """Test model loading failure handling."""
        container = VLLMContainer(model="nonexistent-model", ports={"8001": "8001"})

        with container:
            container.start()

            # Wait for failure
            time.sleep(60)

            # Check that model failed to load
            try:
                response = requests.get(f"{container.get_connection_url()}/health")
                # Should not be healthy
                assert response.status_code != 200
            except Exception:
                # Connection failure is expected for failed model
                pass

    @pytest.mark.containerized
    def test_multiple_models_loading(self):
        """Test loading multiple models in parallel."""
        # Skip VLLM tests for now due to persistent device detection issues in containerized environment
        # pytest.skip("VLLM containerized tests disabled due to device detection issues")

        containers = []

        try:
            # Start multiple containers with different models
            models = [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ]

            for i, model in enumerate(models):
                container = VLLMContainer(
                    model=model, ports={str(8002 + i): str(8002 + i)}
                )
                container.start()
                containers.append(container)

            # Wait for all models to load
            for container in containers:
                max_wait = 600
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    try:
                        response = requests.get(
                            f"{container.get_connection_url()}/health"
                        )
                        if response.status_code == 200:
                            break
                    except Exception:
                        time.sleep(5)
                else:
                    pytest.fail(f"Model {container.model} failed to load")

        finally:
            # Cleanup
            for container in containers:
                container.stop()
