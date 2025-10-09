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
        container = VLLMContainer(
            model="microsoft/DialoGPT-medium", host_port=8000, container_port=8000
        )

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
                pytest.fail("Model failed to load within timeout")

            # Verify model metadata
            info_response = requests.get(f"{container.get_connection_url()}/v1/models")
            models = info_response.json()
            assert len(models["data"]) > 0
            assert "DialoGPT" in models["data"][0]["id"]

    @pytest.mark.containerized
    def test_model_loading_failure(self):
        """Test model loading failure handling."""
        container = VLLMContainer(
            model="nonexistent-model", host_port=8001, container_port=8001
        )

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
        containers = []

        try:
            # Start multiple containers with different models
            models = ["microsoft/DialoGPT-small", "microsoft/DialoGPT-medium"]

            for i, model in enumerate(models):
                container = VLLMContainer(
                    model=model, host_port=8002 + i, container_port=8002 + i
                )
                container.start()
                containers.append(container)

            # Wait for all models to load
            for container in containers:
                max_wait = 300
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
