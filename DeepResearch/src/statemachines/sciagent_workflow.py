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
class Hypothesis:
    """Structured representation of a single hypothesis."""
    id: str
    statement: str
    rationale: str
    prior_confidence: float  # 0.0–1.0


@dataclass
class EvidencePlanItem:
    """Planned evidence source / query."""
    id: str
    source: str          # e.g. "PubMed", "GO", "PDB"
    description: str
    query: str
    priority: int        # 1 = highest


@dataclass
class SciAgentTrace:
    """Full trace for one SciAgent run."""
    question: str
    created_at: str
    config_snapshot: Dict[str, Any]
    steps: List[SciAgentStep]
    hypotheses: List[Hypothesis]
    evidence_plan: List[EvidencePlanItem]
    methods_plan: str
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
          use_deepsearch: true
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
    SciAgent workflow with DeepSearch integration.

    For now this is a lightweight, single-function orchestration that:
      - Builds a structured reasoning trace with placeholder steps
      - Populates structured hypotheses, evidence plan, and methods plan
      - Optionally calls the DeepSearch workflow to gather external evidence
      - Optionally writes the trace as a dataset row
      - Returns a human-readable answer string

    This is intentionally simple and safe; future work can:
      - Integrate PRIME / Bioinformatics tools
      - Use Pydantic Graph for multi-node execution
    """
    sci_cfg = _get_sciagent_cfg(cfg)

    params = sci_cfg.get("params", {}) if isinstance(sci_cfg, dict) else {}
    max_iterations = params.get("max_iterations", 4)
    trace_reasoning = params.get("trace_reasoning", True)
    use_deepsearch = params.get("use_deepsearch", True)

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

    # --- Step 2: Generate structured hypotheses (placeholder content) ---
    hypotheses: List[Hypothesis] = [
        Hypothesis(
            id="H1",
            statement="The observed phenomenon is regulated by multiple interacting pathways.",
            rationale="Many biological and physical systems show emergent behavior arising from pathway interactions.",
            prior_confidence=0.6,
        ),
        Hypothesis(
            id="H2",
            statement="A single dominant mechanism explains most of the observed variance, with context-dependent modifiers.",
            rationale="Often one major factor explains the bulk of the effect size, while other modifiers fine-tune outcomes.",
            prior_confidence=0.4,
        ),
    ]

    steps.append(
        SciAgentStep(
            name="generate_hypotheses",
            description="Generated a small set of candidate hypotheses with rationales and prior confidence.",
            metadata={
                "hypotheses": [asdict(h) for h in hypotheses],
                "max_iterations": max_iterations,
            },
        )
    )

    # --- Step 3: Plan evidence collection (placeholder, but structured) ---
    evidence_plan: List[EvidencePlanItem] = [
        EvidencePlanItem(
            id="E1",
            source="PubMed",
            description="Retrieve primary research articles and recent reviews relevant to the question.",
            query=f'{question} review[pt] OR mechanistic[tiab]',
            priority=1,
        ),
        EvidencePlanItem(
            id="E2",
            source="Preprint servers",
            description="Search preprints for cutting-edge or not-yet-reviewed studies.",
            query=question,
            priority=2,
        ),
        EvidencePlanItem(
            id="E3",
            source="Databases",
            description="Query structured databases (GO, PDB, etc.) if the question involves genes/proteins.",
            query="entity-specific queries to GO / PDB / related resources",
            priority=3,
        ),
    ]

    steps.append(
        SciAgentStep(
            name="plan_evidence_collection",
            description="Outlined a plan to gather evidence via DeepSearch / RAG and structured databases.",
            metadata={
                "evidence_plan": [asdict(e) for e in evidence_plan],
                "use_deepsearch_flow": use_deepsearch,
            },
        )
    )

    # --- Optional Step 4: Call DeepSearch workflow ---
    deepsearch_answer: Optional[str] = None
    if use_deepsearch:
        try:
            # Local import to avoid any potential circular imports at module load time
            from .deepsearch_workflow import run_deepsearch_workflow

            deepsearch_answer = run_deepsearch_workflow(question, cfg)
            steps.append(
                SciAgentStep(
                    name="deepsearch_execution",
                    description="Executed DeepSearch workflow to collect and synthesize external evidence.",
                    metadata={
                        "deepsearch_answer": deepsearch_answer,
                    },
                )
            )
        except Exception as e:
            steps.append(
                SciAgentStep(
                    name="deepsearch_error",
                    description="Attempted to run DeepSearch workflow but it failed.",
                    metadata={
                        "error": str(e),
                    },
                )
            )

    # --- Step 5: Methods-style plan (for SFT/DPO later) ---
    methods_plan = (
        "1. Formulate the research question precisely and identify key entities.\n"
        "2. Generate multiple mechanistic hypotheses with explicit rationales and prior confidence.\n"
        "3. Design an evidence collection strategy using:\n"
        "   - PubMed for primary literature and reviews\n"
        "   - Preprint servers for recent, not-yet-reviewed work\n"
        "   - Structured databases (e.g., GO, PDB) when genes/proteins are involved\n"
        "4. Retrieve and filter evidence based on study quality, recency, and relevance.\n"
        "5. Map each piece of evidence to supporting or refuting specific hypotheses.\n"
        "6. Reweight hypothesis confidence based on the aggregated evidence.\n"
        "7. Synthesize a critical narrative that explains which hypothesis is best supported and why."
    )

    steps.append(
        SciAgentStep(
            name="construct_methods_plan",
            description="Defined a methods-style plan for how SciAgent should collect and evaluate evidence.",
            metadata={
                "methods_plan": methods_plan,
            },
        )
    )

    # --- Step 6: Synthesis (DeepSearch-aware text for now) ---
    if deepsearch_answer:
        synthesized_answer = (
            "SciAgent used the DeepSearch workflow to gather external evidence and produced the "
            "following synthesized summary:\n\n"
            f"{deepsearch_answer}\n\n"
            "In addition, SciAgent constructed structured hypotheses, an evidence collection plan, "
            "and a methods-style evaluation procedure that can be reused for dataset generation "
            "and future tool-grounded reasoning."
        )
    else:
        synthesized_answer = (
            "SciAgent (minimal stub) analyzed your question and constructed a generic scientific "
            "reasoning template. It generated structured hypotheses, an evidence collection plan, "
            "and a methods-style evaluation procedure. Future versions will plug this plan into "
            "tool-grounded flows (DeepSearch, RAG, PRIME, Bioinformatics) to produce fully "
            "verifiable, data-backed conclusions."
        )

    steps.append(
        SciAgentStep(
            name="synthesize_answer",
            description="Synthesized a high-level answer based on the placeholder pipeline.",
            metadata={
                "hypotheses_considered": len(hypotheses),
                "trace_reasoning": trace_reasoning,
                "used_deepsearch": bool(deepsearch_answer),
            },
        )
    )

    trace = SciAgentTrace(
        question=question,
        created_at=datetime.utcnow().isoformat() + "Z",
        config_snapshot=sci_cfg,
        steps=steps if trace_reasoning else [],
        hypotheses=hypotheses,
        evidence_plan=evidence_plan,
        methods_plan=methods_plan,
        final_answer=synthesized_answer,
    )

    # Optional dataset logging
    _maybe_write_dataset(trace, sci_cfg)

    return synthesized_answer

