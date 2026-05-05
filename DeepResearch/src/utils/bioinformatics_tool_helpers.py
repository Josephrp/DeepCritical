"""Helpers for bioinformatics tools when binaries are missing on the host.

Production workflows must fail loudly unless explicit mock mode is enabled.
"""

from __future__ import annotations

import os
import shutil
from typing import Any


def response_if_executable_missing(
    exe_name: str,
    mock_success_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Return ``None`` if ``exe_name`` is on PATH; otherwise mock or error.

    Set ``DEEPC_BIOINFORMATICS_ALLOW_MOCK=1`` to preserve legacy mock-success
    behaviour when dependencies are not installed (development/tests only).
    """
    if shutil.which(exe_name):
        return None
    if os.getenv("DEEPC_BIOINFORMATICS_ALLOW_MOCK", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return mock_success_payload
    return {
        "success": False,
        "error": (
            f"Executable '{exe_name}' not found on PATH. "
            "Install it or set DEEPC_BIOINFORMATICS_ALLOW_MOCK=1 for "
            "development-only mocks."
        ),
    }
