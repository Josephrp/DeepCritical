"""Tests for hypothesis pipeline response shape (quality judge surfacing)."""

from __future__ import annotations

from DeepResearch.src.agents.workflow_orchestrator import (
    _hypothesis_generation_return_payload,
)
from DeepResearch.src.datatypes.workflow_orchestration import HypothesisDataset


def test_hypothesis_payload_surfaces_failed_quality_judge() -> None:
    ds = HypothesisDataset(
        name="batch",
        description="d",
        hypotheses=[{"statement": "Cold stress alters membrane fluidity."}],
    )
    judge = {
        "success": False,
        "judge_id": "hypothesis_quality_judge",
        "overall_score": 0.0,
        "criterion_scores": {},
        "feedback": "",
        "recommendations": [],
        "error_message": "model timeout",
    }
    out = _hypothesis_generation_return_payload(
        dataset=ds,
        meta={"steps": 1},
        judge_payload=judge,
        run_quality_judge=True,
    )
    assert out["success"] is True
    assert out["quality_judge_ok"] is False
    assert "timeout" in (out.get("quality_judge_error") or "")


def test_hypothesis_payload_omits_quality_fields_when_judge_disabled() -> None:
    ds = HypothesisDataset(
        name="b",
        description="d",
        hypotheses=[{"statement": "y"}],
    )
    out = _hypothesis_generation_return_payload(
        dataset=ds,
        meta={},
        judge_payload=None,
        run_quality_judge=False,
    )
    assert "quality_judge_ok" not in out
    assert "quality_judge_error" not in out
