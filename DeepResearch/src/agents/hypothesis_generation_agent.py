"""
LLM pipeline: evidence-grounded hypothesis proposal, optional critic, dedupe, dataset build.
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from DeepResearch.src.datatypes.hypothesis_generation import (
    HypothesisCandidate,
    HypothesisGenerationParams,
    build_hypothesis_dataset,
    dedupe_and_cap,
)
from DeepResearch.src.datatypes.workflow_orchestration import HypothesisDataset
from DeepResearch.src.prompts.hypothesis_generation import (
    SYSTEM_CRITIC,
    SYSTEM_PROPOSER,
    build_critic_user_prompt,
    build_proposer_user_prompt,
)
from DeepResearch.src.utils.hypothesis_evidence import fetch_hypothesis_evidence
from DeepResearch.src.utils.model_registry import (
    resolve_model_name,
    resolve_pydantic_ai_model,
)


class HypothesisBatchOutput(BaseModel):
    hypotheses: list[HypothesisCandidate] = Field(default_factory=list)


class CritiqueItem(BaseModel):
    index: int = Field(ge=0)
    keep: bool = True
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class CritiqueBatch(BaseModel):
    critiques: list[CritiqueItem] = Field(default_factory=list)


def _extract_question(input_data: dict[str, Any]) -> str:
    for key in ("question", "task_description", "research_brief", "query"):
        v = input_data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _attach_global_evidence(
    candidates: list[HypothesisCandidate],
    global_refs: list[Any],
    max_refs_per: int = 3,
) -> None:
    """Attach top evidence refs to each candidate when it has none."""
    if not global_refs:
        return
    for c in candidates:
        if not c.evidence:
            c.evidence = global_refs[:max_refs_per]


def _apply_critiques(
    hypotheses: list[HypothesisCandidate],
    batch: CritiqueBatch,
) -> list[HypothesisCandidate]:
    if not batch.critiques:
        return hypotheses
    drops = {c.index for c in batch.critiques if not c.keep}
    conf_overrides = {c.index: c.confidence for c in batch.critiques if c.keep}
    out: list[HypothesisCandidate] = []
    for i, h in enumerate(hypotheses):
        if i in drops:
            continue
        conf = conf_overrides.get(i, h.confidence)
        out.append(h.model_copy(update={"confidence": conf}))
    return out if out else hypotheses


def _pydantic_model_for_hypothesis(
    params: HypothesisGenerationParams,
) -> str | object:
    """Use registry resolution when params still reference the default chat model."""
    default_name = resolve_model_name(None, "default")
    if params.model_name == default_name:
        return resolve_pydantic_ai_model(None, "default")
    return params.model_name


async def _run_critic(
    model: str | object,
    question: str,
    evidence_context: str,
    hypotheses: list[HypothesisCandidate],
) -> CritiqueBatch:
    agent = Agent(
        model=model,
        output_type=CritiqueBatch,
        system_prompt=SYSTEM_CRITIC,
    )
    hy_payload = [h.model_dump(mode="json") for h in hypotheses]
    prompt = build_critic_user_prompt(
        question,
        evidence_context,
        json.dumps(hy_payload, indent=2, default=str),
    )
    run = await agent.run(prompt)
    out = run.output
    assert isinstance(out, CritiqueBatch)
    return out


async def run_hypothesis_generation_pipeline(
    input_data: dict[str, Any],
    parameters: dict[str, Any],
    *,
    default_model: str | None = None,
) -> tuple[HypothesisDataset, dict[str, Any]]:
    """
    Full async pipeline: evidence → propose → optional critic → dedupe → HypothesisDataset.

    Returns:
        dataset, metadata (timing, flags, evidence info).
    """
    dm = default_model or resolve_model_name(None, "default")
    params = HypothesisGenerationParams.from_mapping(parameters, default_model=dm)
    question = _extract_question(input_data)
    if not question:
        msg = (
            "Missing question, task_description, research_brief, or query in input_data"
        )
        raise ValueError(msg)

    t0 = time.perf_counter()
    meta: dict[str, Any] = {
        "truncated": False,
        "evidence_degraded": False,
        "chunk_count": 0,
    }

    evidence_text, evidence_refs, ev_meta = await fetch_hypothesis_evidence(
        question, params
    )
    meta.update(ev_meta)

    proposer = Agent(
        model=_pydantic_model_for_hypothesis(params),
        output_type=HypothesisBatchOutput,
        system_prompt=SYSTEM_PROPOSER,
    )
    user_prompt = build_proposer_user_prompt(
        question,
        evidence_text,
        params.hypothesis_types,
        params.max_hypotheses,
    )
    run = await proposer.run(
        user_prompt,
        model_settings=ModelSettings(temperature=params.temperature),
    )
    bout = run.output
    assert isinstance(bout, HypothesisBatchOutput)
    candidates = list(bout.hypotheses)
    _attach_global_evidence(candidates, evidence_refs)

    if params.enable_critic_pass and candidates:
        critique = await _run_critic(
            _pydantic_model_for_hypothesis(params),
            question,
            evidence_text,
            candidates,
        )
        candidates = _apply_critiques(candidates, critique)

    candidates, truncated = dedupe_and_cap(candidates, params.max_hypotheses)
    meta["truncated"] = truncated
    meta["elapsed_seconds"] = time.perf_counter() - t0
    meta["hypothesis_count"] = len(candidates)

    wf_name = str(input_data.get("workflow_name", "hypothesis_generation"))
    ds_name = str(input_data.get("dataset_name", "Generated hypotheses"))
    ds_desc = str(
        input_data.get(
            "dataset_description",
            f"Evidence-grounded hypotheses for: {question[:200]}",
        )
    )
    dataset = build_hypothesis_dataset(
        ds_name,
        ds_desc,
        candidates,
        [wf_name],
        extra_metadata=meta,
    )

    return dataset, meta


def summarize_hypotheses_for_answer(candidates: list[HypothesisCandidate]) -> str:
    """Short text summary for evaluator / primary workflow."""
    lines: list[str] = []
    for i, h in enumerate(candidates, start=1):
        lines.append(
            f"{i}. [{h.hypothesis_type.value}] {h.statement} (confidence={h.confidence:.2f})"
        )
    return "\n".join(lines) if lines else "No hypotheses generated."
