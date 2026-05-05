"""
Typed models for evidence-grounded hypothesis generation workflows.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from DeepResearch.src.datatypes.workflow_orchestration import HypothesisDataset
from DeepResearch.src.utils.model_registry import resolve_model_name


class HypothesisType(str, Enum):
    """Categories aligned with workflow orchestration YAML."""

    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    MECHANISTIC = "mechanistic"
    UNKNOWN = "unknown"


class EvidenceRef(BaseModel):
    """A single piece of evidence supporting or contextualizing hypotheses."""

    source_type: str = Field("web", description="Origin: web, document, rag, etc.")
    title: str = ""
    url: str = ""
    snippet: str = ""
    chunk_id: str | None = None

    model_config = ConfigDict(extra="ignore")


class HypothesisCandidate(BaseModel):
    """One structured, testable hypothesis proposal."""

    statement: str = Field(..., min_length=1, description="Core hypothesis statement")
    hypothesis_type: HypothesisType = HypothesisType.UNKNOWN
    mechanism_or_rationale: str = ""
    predictions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("hypothesis_type", mode="before")
    @classmethod
    def coerce_hypothesis_type(cls, v: Any) -> Any:
        if isinstance(v, HypothesisType):
            return v
        if isinstance(v, str):
            key = v.strip().lower().replace(" ", "_")
            try:
                return HypothesisType(key)
            except ValueError:
                return HypothesisType.UNKNOWN
        return HypothesisType.UNKNOWN

    def to_dataset_entry(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class HypothesisGenerationParams(BaseModel):
    """Sub-workflow parameters for hypothesis generation (Hydra-compatible)."""

    hypothesis_types: list[str] = Field(
        default_factory=lambda: ["causal", "correlational", "mechanistic"],
    )
    enable_evidence_synthesis: bool = True
    enable_confidence_scoring: bool = True
    enable_critic_pass: bool = False
    max_hypotheses: int = Field(10, ge=1, le=50)
    max_evidence_chars: int = Field(12000, ge=1000, le=100000)
    model_name: str = Field(
        default_factory=lambda: resolve_model_name(None, "default"),
    )
    temperature: float = Field(0.4, ge=0.0, le=2.0)
    num_results: int = Field(4, ge=1, le=20)
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(0, ge=0, le=2000)
    search_type: str = "search"
    extra_document_text: str = ""
    run_quality_judge: bool = False
    judge_id: str = "hypothesis_quality_judge"

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        default_model: str | None = None,
    ) -> HypothesisGenerationParams:
        """Build params from orchestration ``parameters`` dict with safe defaults."""
        dm = default_model or resolve_model_name(None, "default")
        base: dict[str, Any] = {
            "hypothesis_types": data.get(
                "hypothesis_types",
                ["causal", "correlational", "mechanistic"],
            ),
            "enable_evidence_synthesis": data.get("enable_evidence_synthesis", True),
            "enable_confidence_scoring": data.get("enable_confidence_scoring", True),
            "enable_critic_pass": data.get("enable_critic_pass", False),
            "max_hypotheses": int(data.get("max_hypotheses", 10)),
            "max_evidence_chars": int(data.get("max_evidence_chars", 12000)),
            "model_name": str(data.get("model_name", dm)),
            "temperature": float(data.get("temperature", 0.4)),
            "num_results": int(data.get("num_results", 4)),
            "chunk_size": int(data.get("chunk_size", 1000)),
            "chunk_overlap": int(data.get("chunk_overlap", 0)),
            "search_type": str(data.get("search_type", "search")),
            "extra_document_text": str(data.get("extra_document_text", "")),
            "run_quality_judge": bool(data.get("run_quality_judge", False)),
            "judge_id": str(data.get("judge_id", "hypothesis_quality_judge")),
        }
        return cls.model_validate(base)


def build_hypothesis_dataset(
    name: str,
    description: str,
    candidates: list[HypothesisCandidate],
    source_workflows: list[str],
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> HypothesisDataset:
    """Construct a :class:`HypothesisDataset` from structured candidates."""
    meta = dict(extra_metadata or {})
    return HypothesisDataset(
        name=name,
        description=description,
        hypotheses=[c.to_dataset_entry() for c in candidates],
        metadata=meta,
        source_workflows=list(source_workflows),
    )


def normalize_statement_for_dedupe(statement: str) -> str:
    """Normalize hypothesis text for near-duplicate detection (stdlib only)."""
    s = statement.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s[:500]


def dedupe_and_cap(
    candidates: list[HypothesisCandidate],
    max_hypotheses: int,
) -> tuple[list[HypothesisCandidate], bool]:
    """Drop near-duplicates and cap count; sort by confidence descending."""
    sorted_c = sorted(candidates, key=lambda x: x.confidence, reverse=True)
    seen: set[str] = set()
    out: list[HypothesisCandidate] = []
    for c in sorted_c:
        key = normalize_statement_for_dedupe(c.statement)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_hypotheses:
            return out, len(sorted_c) > max_hypotheses
    return out, False
