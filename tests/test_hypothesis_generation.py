"""Tests for hypothesis generation datatypes, evidence layer, and workflow wiring."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from DeepResearch.src.agents.workflow_orchestrator import PrimaryWorkflowOrchestrator
from DeepResearch.src.datatypes.hypothesis_generation import (
    HypothesisCandidate,
    HypothesisGenerationParams,
    HypothesisType,
    build_hypothesis_dataset,
    dedupe_and_cap,
    normalize_statement_for_dedupe,
)
from DeepResearch.src.datatypes.llm_models import DEFAULT_PYDANTIC_AI_MODEL
from DeepResearch.src.datatypes.workflow_orchestration import (
    HypothesisDataset,
    WorkflowConfig,
    WorkflowOrchestrationConfig,
    WorkflowType,
)


def test_hypothesis_generation_params_from_mapping() -> None:
    p = HypothesisGenerationParams.from_mapping(
        {"max_hypotheses": 3, "enable_critic_pass": True},
        default_model="openai:gpt-4o-mini",
    )
    assert p.max_hypotheses == 3
    assert p.enable_critic_pass is True
    assert p.model_name == "openai:gpt-4o-mini"


def test_hypothesis_candidate_coerce_type() -> None:
    h = HypothesisCandidate.model_validate(
        {"statement": "S", "hypothesis_type": "causal"}
    )
    assert h.hypothesis_type == HypothesisType.CAUSAL
    h2 = HypothesisCandidate.model_validate(
        {"statement": "S", "hypothesis_type": "not_a_real_type"}
    )
    assert h2.hypothesis_type == HypothesisType.UNKNOWN


def test_dedupe_and_cap() -> None:
    a = HypothesisCandidate(
        statement="Alpha drives beta",
        confidence=0.9,
        hypothesis_type=HypothesisType.CAUSAL,
    )
    b = HypothesisCandidate(
        statement="alpha drives beta",
        confidence=0.5,
        hypothesis_type=HypothesisType.CAUSAL,
    )
    c = HypothesisCandidate(
        statement="Gamma correlates with delta",
        confidence=0.7,
        hypothesis_type=HypothesisType.CORRELATIONAL,
    )
    out, truncated = dedupe_and_cap([a, b, c], max_hypotheses=2)
    assert len(out) == 2
    assert truncated
    assert out[0].statement == "Alpha drives beta"


def test_normalize_statement() -> None:
    assert "alpha" in normalize_statement_for_dedupe("  Alpha   beta  ")


def test_build_hypothesis_dataset_roundtrip() -> None:
    c = HypothesisCandidate(
        statement="X",
        hypothesis_type=HypothesisType.MECHANISTIC,
        confidence=0.8,
    )
    ds = build_hypothesis_dataset("n", "d", [c], ["wf"])
    assert isinstance(ds, HypothesisDataset)
    assert len(ds.hypotheses) == 1
    assert ds.hypotheses[0]["statement"] == "X"


@pytest.mark.asyncio
async def test_fetch_hypothesis_evidence_degraded() -> None:
    from DeepResearch.src.utils import hypothesis_evidence as he

    p = HypothesisGenerationParams(
        enable_evidence_synthesis=True,
        max_evidence_chars=8000,
    )
    mock_res = MagicMock()
    mock_res.success = False
    mock_res.error = "network"
    mock_res.data = {}
    with patch.object(he.canonical_registry, "aexecute", new_callable=AsyncMock) as m:
        m.return_value = mock_res
        text, _refs, meta = await he.fetch_hypothesis_evidence("q", p)
    assert meta["evidence_degraded"] is True
    assert text == ""


@pytest.mark.asyncio
async def test_execute_hypothesis_generation_workflow_mocked_pipeline() -> None:
    async def fake_run_hypothesis_workflow(
        question: str,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        c = HypothesisCandidate(
            statement="Mock hypothesis",
            hypothesis_type=HypothesisType.CAUSAL,
            confidence=0.88,
        )
        ds = build_hypothesis_dataset(
            "batch",
            "desc",
            [c],
            [""],
        )
        return {
            "success": True,
            "hypothesis_dataset": ds,
            "hypotheses": [{"statement": "Mock hypothesis"}],
            "answer": "Summary with Mock hypothesis",
        }

    cfg = WorkflowOrchestrationConfig(
        primary_workflow=WorkflowConfig(
            workflow_type=WorkflowType.PRIMARY_REACT,
            name="p",
            parameters={"model_name": DEFAULT_PYDANTIC_AI_MODEL},
        ),
    )
    with (
        patch(
            "DeepResearch.src.agents.workflow_orchestrator.Agent",
            return_value=MagicMock(),
        ),
        patch(
            "DeepResearch.src.statemachines.hypothesis_workflow.run_hypothesis_workflow",
            new=fake_run_hypothesis_workflow,
        ),
    ):
        orch = PrimaryWorkflowOrchestrator(cfg)
        out = await orch._execute_hypothesis_generation_workflow(
            {"question": "Why is the sky blue?"},
            {"max_hypotheses": 5},
        )
    assert out["success"] is True
    assert "hypothesis_dataset" in out
    assert out["hypotheses"][0]["statement"] == "Mock hypothesis"
    assert "Mock hypothesis" in out["answer"]


@pytest.mark.asyncio
async def test_hypothesis_tool_runner_validate_defaults() -> None:
    from DeepResearch.src.tools.hypothesis_generation_tool import (
        HypothesisGenerationToolRunner,
    )

    r = HypothesisGenerationToolRunner()
    ok, err = r.validate({"question": "q"})
    assert ok is True
    assert err is None
