"""LLM-backed judge execution for workflow orchestration (MVP)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from DeepResearch.src.datatypes.workflow_orchestration import (
    JudgeConfig,
    JudgeEvaluationRequest,
    JudgeEvaluationResult,
)


class JudgeLLMOutput(BaseModel):
    """Structured rubric output from the judge model."""

    overall_score: float = Field(ge=0.0, le=10.0)
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    recommendations: list[str] = Field(default_factory=list)


_JUDGE_SYSTEM_SUPPLEMENT: dict[str, str] = {
    "hypothesis_quality_judge": (
        "The content may be a hypothesis_dataset with multiple structured hypotheses. "
        "Score testability, falsifiability, specificity, evidence linkage, and internal "
        "consistency across the batch. Penalize vague or non-falsifiable statements."
    ),
}


async def run_llm_judge(
    judge_config: JudgeConfig,
    request: JudgeEvaluationRequest,
) -> JudgeEvaluationResult:
    """Run a rubric judge using the configured model and return structured scores."""
    base = (
        "You are an expert evaluator. Score the given content against the "
        "criteria using the output schema. Scores are on the scale described "
        "in the user message (typically 1-10 per criterion and overall)."
    )
    extra = _JUDGE_SYSTEM_SUPPLEMENT.get(request.judge_id, "")
    system_prompt = f"{base}\n\n{extra}" if extra else base
    agent = Agent(
        model=judge_config.model_name,
        output_type=JudgeLLMOutput,
        system_prompt=system_prompt,
    )
    prompt = (
        f"Judge ID: {request.judge_id}\n"
        f"Judge name: {judge_config.name}\n"
        f"Evaluation criteria: {request.evaluation_criteria}\n"
        f"Scoring scale: {judge_config.scoring_scale}\n\n"
        f"Content to evaluate (JSON):\n{json.dumps(request.content_to_evaluate, default=str)}\n\n"
        f"Additional context (JSON):\n{json.dumps(request.context, default=str)}\n"
    )
    try:
        run = await agent.run(prompt)
        out = run.output
        assert isinstance(out, JudgeLLMOutput)
        return JudgeEvaluationResult(
            success=True,
            judge_id=request.judge_id,
            overall_score=out.overall_score,
            criterion_scores=out.criterion_scores,
            feedback=out.feedback,
            recommendations=out.recommendations,
        )
    except Exception as e:
        return JudgeEvaluationResult(
            success=False,
            judge_id=request.judge_id,
            overall_score=0.0,
            criterion_scores={},
            feedback="",
            recommendations=[],
            error_message=str(e),
        )
