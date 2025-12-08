from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig


@dataclass
class SciAgentStep:
    """Single step in the SciAgent reasoning trace."""
    name: str
    description: str
    metadata: Dict[str, Any]


@dataclass
class SciAgentTrace:
    """Full trace for one SciAgent run."""
    question: str
    created_at: str
    config_snapshot: Dict[str, Any]
    steps: List[SciAgentStep]
    final_answer: str


def _get_sciagent_cfg(cfg: Optional[DictConfig]) -> Dict[str, Any]:
    """
    Safely extract the SciAgent config block from the global Hydra config.

    Expected structure (in configs/config.yaml):

    flows:
      sciagent:
        enabled: true
        params:
          max_iterations: 8
          trace_reasoning: true
          generate_datasets: true
          require_verifiable_sources: true
    """
    if cfg is None:
        return {}

    flows_cfg = getattr(cfg, "flows", None)
    sciagent_cfg = getattr(flows_cfg, "sciagent", None)
    if sciagent_cfg is None:
        return {}

    # Convert to plain dict to avoid leaking OmegaConf objects around
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(sciagent_cfg, resolve=True)  # type: ignore
    except Exception:
        # Fallback: best-effort conversion
        return dict(sciagent_cfg)


def _maybe_write_dataset(trace: SciAgentTrace, sci_cfg: Dict[str, Any]) -> None:
    """
    Optionally write a JSONL dataset row for this run.

    Controlled by flows.sciagent.params.generate_datasets.
    """
    params = sci_cfg.get("params", {}) if isinstance(sci_cfg, dict) else {}
    generate = params.get("generate_datasets", False)
    if not generate:
        return

    out_dir = os.path.join("outputs", "datasets", "sciagent")
    os.makedirs(out_dir, exist_ok=True)

    # Simple filename with timestamp
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(out_dir, f"sciagent_{ts}.jsonl")

    payload = asdict(trace)
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_sciagent_workflow(question: str, cfg: Optional[DictConfig] = None) -> str:
    """
    Minimal SciAgent workflow.

    For now this is a lightweight, single-function orchestration that:
      - Builds a structured reasoning trace with placeholder steps
      - Optionally writes the trace as a dataset row
      - Returns a human-readable answer string

    This is intentionally simple and safe; future work can:
      - Call DeepSearch / RAG flows
      - Integrate PRIME / Bioinformatics tools
      - Use Pydantic Graph for multi-node execution
    """
    sci_cfg = _get_sciagent_cfg(cfg)

    params = sci_cfg.get("params", {}) if isinstance(sci_cfg, dict) else {}
    max_iterations = params.get("max_iterations", 4)
    trace_reasoning = params.get("trace_reasoning", True)

    steps: List[SciAgentStep] = []

    # --- Step 1: Parse ---
    steps.append(
        SciAgentStep(
            name="parse_question",
            description="Parsed the research question and identified key entities and intent.",
            metadata={
                "question": question,
                "detected_intent": "general_scientific_reasoning",
                "entities": [],  # placeholder for future NER / ontology mapping
            },
        )
    )

    # --- Step 2: Generate hypotheses (placeholder) ---
    hypotheses = [
        "H1: The observed phenomenon is regulated by multiple interacting pathways.",
        "H2: There exists a dominant mechanism with context-dependent modifiers.",
    ]
    steps.append(
        SciAgentStep(
            name="generate_hypotheses",
            description="Generated a small set of candidate hypotheses.",
            metadata={
                "hypotheses": hypotheses,
                "max_iterations": max_iterations,
            },
        )
    )

    # --- Step 3: Literature / evidence planning (placeholder) ---
    steps.append(
        SciAgentStep(
            name="plan_evidence_collection",
            description="Outlined a plan to gather evidence via DeepSearch / RAG flows.",
            metadata={
                "planned_sources": [
                    "PubMed",
                    "preprint servers",
                    "review articles",
                    "databases (GO, PDB, etc.)",
                ],
                "use_deepsearch_flow": True,
                "use_rag_flow": True,
            },
        )
    )

    # --- Step 4: Synthesis (placeholder) ---
    synthesized_answer = (
        "SciAgent (minimal stub) analyzed your question and constructed a "
        "generic scientific reasoning template. The current implementation "
        "does not yet execute full DeepSearch / PRIME / Bioinformatics flows, "
        "but it prepares structured hypotheses and an evidence-collection plan "
        "that future versions will use for verifiable, tool-grounded reasoning."
    )
    steps.append(
        SciAgentStep(
            name="synthesize_answer",
            description="Synthesized a high-level answer based on the placeholder pipeline.",
            metadata={
                "hypotheses_considered": len(hypotheses),
                "trace_reasoning": trace_reasoning,
            },
        )
    )

    trace = SciAgentTrace(
        question=question,
        created_at=datetime.utcnow().isoformat() + "Z",
        config_snapshot=sci_cfg,
        steps=steps if trace_reasoning else [],
        final_answer=synthesized_answer,
    )

    # Optional dataset logging
    _maybe_write_dataset(trace, sci_cfg)

    return synthesized_answer

