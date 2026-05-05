"""Prompts for evidence-grounded hypothesis generation."""

from __future__ import annotations

SYSTEM_PROPOSER = """You are a senior research scientist generating testable hypotheses.

Rules:
- Every hypothesis must be falsifiable and specific enough to design an experiment or analysis.
- Ground statements in the evidence context when provided; if evidence is thin, say so in limitations.
- Prefer clarity over novelty; avoid vague claims.
- Assign hypothesis_type one of: causal, correlational, mechanistic, or unknown if none fits.
- confidence is 0.0–1.0 reflecting strength of evidence and specificity.
- predictions are concrete observable outcomes that would support the hypothesis.
- limitations must list key assumptions or missing evidence.

Output must match the structured schema exactly (no extra prose outside structured fields)."""

SYSTEM_CRITIC = """You are a rigorous scientific critic reviewing a batch of hypotheses.

For each hypothesis index (0-based), decide whether to keep it and assign an updated confidence in [0,1].
Drop hypotheses that are unfalsifiable, duplicate others, or unsupported by the evidence context.
Return structured output only."""


def build_proposer_user_prompt(
    question: str,
    evidence_context: str,
    hypothesis_types: list[str],
    max_hypotheses: int,
) -> str:
    types_line = ", ".join(hypothesis_types)
    ev = (
        evidence_context.strip()
        or "(No external evidence retrieved; rely on general scientific reasoning and flag limitations.)"
    )
    return (
        f"Research focus / question:\n{question}\n\n"
        f"Target hypothesis types to cover where appropriate: {types_line}\n"
        f"Produce at most {max_hypotheses} distinct hypotheses.\n\n"
        f"Evidence context:\n{ev}\n"
    )


def build_critic_user_prompt(
    question: str,
    evidence_context: str,
    hypotheses_json: str,
) -> str:
    return (
        f"Original question:\n{question}\n\n"
        f"Evidence context (for grounding):\n{evidence_context.strip() or '(none)'}\n\n"
        f"Hypotheses (JSON array of objects with at least statement, hypothesis_type, confidence):\n"
        f"{hypotheses_json}\n"
    )
