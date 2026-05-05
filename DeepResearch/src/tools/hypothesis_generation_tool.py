"""
Canonical tool: evidence-grounded hypothesis generation pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from DeepResearch.src.agents.hypothesis_generation_agent import (
    run_hypothesis_generation_pipeline,
)
from DeepResearch.src.tools.base import ExecutionResult, ToolRunner, ToolSpec


class HypothesisGenerationToolRunner(ToolRunner):
    """Run :func:`run_hypothesis_generation_pipeline` (sync wrapper for registry)."""

    def __init__(self) -> None:
        spec = ToolSpec(
            name="hypothesis_generation",
            description=(
                "Generate evidence-grounded, structured scientific hypotheses "
                "(web retrieval + LLM proposer + optional critic)."
            ),
            inputs={
                "question": "TEXT",
                "parameters": "TEXT",
            },
            outputs={
                "hypothesis_dataset": "JSON",
                "metadata": "JSON",
                "success": "BOOLEAN",
                "error": "TEXT",
            },
        )
        super().__init__(spec)

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        merged = dict(params)
        if "parameters" not in merged:
            merged["parameters"] = "{}"
        return ToolRunner.validate(self, merged)

    def run(self, params: dict[str, Any]) -> ExecutionResult:
        params = dict(params)
        if "parameters" not in params:
            params["parameters"] = "{}"
        question = str(params.get("question", "")).strip()
        if not question:
            return ExecutionResult(success=False, error="question is required")
        raw = params.get("parameters", "{}")
        if isinstance(raw, dict):
            parameters = dict(raw)
        else:
            try:
                parameters = json.loads(str(raw)) if str(raw).strip() else {}
            except json.JSONDecodeError as e:
                return ExecutionResult(
                    success=False,
                    error=f"Invalid JSON in parameters: {e!s}",
                )
        input_data = {
            "question": question,
            "workflow_name": str(
                params.get("workflow_name", "hypothesis_generation_tool")
            ),
            "dataset_name": str(params.get("dataset_name", "tool_hypotheses")),
            "dataset_description": str(
                params.get("dataset_description", question[:500]),
            ),
        }
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            dataset, meta = loop.run_until_complete(
                run_hypothesis_generation_pipeline(
                    input_data,
                    parameters,
                    default_model=params.get("model_name"),
                )
            )
        except Exception as e:
            return ExecutionResult(success=False, error=str(e))
        finally:
            loop.close()
        return ExecutionResult(
            success=True,
            data={
                "hypothesis_dataset": dataset.model_dump(mode="json"),
                "metadata": meta,
                "success": True,
                "error": None,
            },
        )


def register_hypothesis_generation_tool() -> None:
    from DeepResearch.src.tools.base import registry

    registry.register("hypothesis_generation", HypothesisGenerationToolRunner)


register_hypothesis_generation_tool()
