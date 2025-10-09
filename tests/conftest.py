"""
Global pytest configuration for DeepCritical testing framework.
"""

import os
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "containerized: Tests requiring containers")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "bioinformatics: Bioinformatics-specific tests")
    config.addinivalue_line("markers", "llm: LLM framework tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment and markers."""
    # Skip containerized tests if not in CI or if DOCKER_TESTS not set
    if not os.getenv("CI") and not os.getenv("DOCKER_TESTS"):
        skip_containerized = pytest.mark.skip(reason="Containerized tests disabled")
        for item in items:
            if "containerized" in item.keywords:
                item.add_marker(skip_containerized)


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "docker_enabled": os.getenv("DOCKER_TESTS", "false").lower() == "true",
        "performance_enabled": os.getenv("PERFORMANCE_TESTS", "false").lower()
        == "true",
        "integration_enabled": os.getenv("INTEGRATION_TESTS", "true").lower() == "true",
        "test_data_dir": Path(__file__).parent / "test_data",
        "artifacts_dir": Path(__file__).parent.parent / "test_artifacts",
    }
