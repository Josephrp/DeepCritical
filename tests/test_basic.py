"""
Basic tests to verify the testing framework is working.
"""

import pytest


@pytest.mark.unit
def test_basic_assertion():
    """Basic test to verify pytest is working."""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_string_operations():
    """Test string operations."""
    result = "hello world".title()
    assert result == "Hello World"


@pytest.mark.integration
def test_environment_variables():
    """Test that environment variables work."""
    import os

    test_var = os.getenv("TEST_VAR", "default")
    assert test_var == "default"  # Should be default since we didn't set it
