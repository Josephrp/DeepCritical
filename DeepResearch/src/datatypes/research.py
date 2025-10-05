"""
Research workflow data types for DeepCritical's research agent operations.

This module defines data structures for research workflow execution including
step results, research outcomes, and related workflow components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class StepResult:
    """Result of a single research step."""

    action: str
    payload: Dict[str, Any]


@dataclass
class ResearchOutcome:
    """Outcome of a research workflow execution."""

    answer: str
    references: List[str]
    context: Dict[str, Any]
