"""Tests for bioinformatics executable-missing policy."""

from __future__ import annotations

from unittest.mock import patch

from DeepResearch.src.utils.bioinformatics_tool_helpers import (
    response_if_executable_missing,
)


def test_missing_executable_errors_without_mock_env(monkeypatch):
    monkeypatch.delenv("DEEPC_BIOINFORMATICS_ALLOW_MOCK", raising=False)
    with patch(
        "DeepResearch.src.utils.bioinformatics_tool_helpers.shutil.which",
        return_value=None,
    ):
        r = response_if_executable_missing("bcftools", {"success": True})
        assert r is not None
        assert r["success"] is False
        assert "bcftools" in r["error"]


def test_missing_executable_returns_mock_when_flag_set(monkeypatch):
    monkeypatch.setenv("DEEPC_BIOINFORMATICS_ALLOW_MOCK", "1")
    mock_payload = {"success": True, "mock": True}
    with patch(
        "DeepResearch.src.utils.bioinformatics_tool_helpers.shutil.which",
        return_value=None,
    ):
        r = response_if_executable_missing("bcftools", mock_payload)
        assert r == mock_payload


def test_executable_on_path_returns_none():
    with patch(
        "DeepResearch.src.utils.bioinformatics_tool_helpers.shutil.which",
        return_value="/usr/bin/bcftools",
    ):
        assert response_if_executable_missing("bcftools", {"success": True}) is None


def test_allow_mock_truthy_strings(monkeypatch):
    for val in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("DEEPC_BIOINFORMATICS_ALLOW_MOCK", val)
        mock_payload = {"ok": True}
        with patch(
            "DeepResearch.src.utils.bioinformatics_tool_helpers.shutil.which",
            return_value=None,
        ):
            assert response_if_executable_missing("x", mock_payload) == mock_payload
        monkeypatch.delenv("DEEPC_BIOINFORMATICS_ALLOW_MOCK", raising=False)
