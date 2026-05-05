"""Shared fixtures for bioinformatics server tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bioinformatics_allow_mock_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host CI often lacks bioinformatics binaries; allow legacy mock-success in tests."""
    monkeypatch.setenv("DEEPC_BIOINFORMATICS_ALLOW_MOCK", "1")
