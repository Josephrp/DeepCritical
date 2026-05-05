"""
Evidence retrieval for hypothesis generation (web chunks + optional extra text).
"""

from __future__ import annotations

import logging
from typing import Any

from DeepResearch.src.datatypes.hypothesis_generation import (
    EvidenceRef,
    HypothesisGenerationParams,
)
from DeepResearch.src.tools.registry import canonical_registry

logger = logging.getLogger(__name__)


def _ensure_chunked_search_registered() -> None:
    """Import side effect: register ``chunked_search`` with the canonical registry."""
    import DeepResearch.src.tools.websearch_tools


def _chunks_to_refs_and_text(
    chunks: list[Any],
    max_chars: int,
) -> tuple[str, list[EvidenceRef]]:
    """Flatten chunk dicts into context text and structured refs."""
    refs: list[EvidenceRef] = []
    parts: list[str] = []
    used = 0
    for i, c in enumerate(chunks):
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or c.get("content") or "").strip()
        title = str(c.get("title") or c.get("source") or f"chunk_{i}")
        url = str(c.get("url") or "")
        snippet = text[:800] if text else ""
        cid = str(c.get("id") or c.get("chunk_id") or f"chunk_{i}")
        refs.append(
            EvidenceRef(
                source_type="web",
                title=title,
                url=url,
                snippet=snippet,
                chunk_id=cid,
            )
        )
        block = f"[{i + 1}] {title}\n{text}\n"
        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 100:
                parts.append(block[:remain])
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts), refs


async def fetch_hypothesis_evidence(
    query: str,
    params: HypothesisGenerationParams,
) -> tuple[str, list[EvidenceRef], dict[str, Any]]:
    """
    Retrieve evidence: optional web chunked search plus optional ``extra_document_text``.

    Returns:
        context_text, evidence_refs, metadata (includes ``evidence_degraded`` when tools fail).
    """
    meta: dict[str, Any] = {"evidence_degraded": False, "chunk_count": 0}
    refs: list[EvidenceRef] = []
    sections: list[str] = []

    if params.extra_document_text.strip():
        doc = params.extra_document_text.strip()
        cap = min(len(doc), params.max_evidence_chars // 2)
        sections.append(f"## User-provided documents\n{doc[:cap]}")
        refs.append(
            EvidenceRef(
                source_type="document",
                title="user_supplied_context",
                url="",
                snippet=doc[:500],
            )
        )

    if not params.enable_evidence_synthesis:
        text = "\n\n".join(sections) if sections else ""
        return text, refs, meta

    _ensure_chunked_search_registered()
    exec_params = {
        "query": query,
        "search_type": params.search_type,
        "num_results": params.num_results,
        "chunk_size": params.chunk_size,
        "chunk_overlap": params.chunk_overlap,
        "heading_level": 3,
        "min_characters_per_chunk": 50,
        "max_characters_per_section": 4000,
        "clean_text": True,
    }
    try:
        res = await canonical_registry.aexecute("chunked_search", exec_params)
        if not res.success:
            meta["evidence_degraded"] = True
            meta["evidence_error"] = res.error or "chunked_search failed"
            logger.warning("chunked_search failed: %s", meta["evidence_error"])
        else:
            raw = res.data.get("chunks", [])
            chunks = raw if isinstance(raw, list) else []
            budget = params.max_evidence_chars - sum(len(s) for s in sections)
            budget = max(budget, 2000)
            ctx_chunk, chunk_refs = _chunks_to_refs_and_text(chunks, budget)
            meta["chunk_count"] = len(chunk_refs)
            if ctx_chunk:
                sections.append("## Web evidence (chunked search)\n" + ctx_chunk)
            refs.extend(chunk_refs)
    except Exception as e:
        meta["evidence_degraded"] = True
        meta["evidence_error"] = str(e)
        logger.exception("Evidence fetch raised")

    text = "\n\n".join(s for s in sections if s).strip()
    if not text:
        meta["evidence_degraded"] = True
    return text, refs, meta
