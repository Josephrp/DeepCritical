from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Orchestrator:
    """Placeholder orchestrator that would sequence subflows based on config."""

    def build_plan(self, question: str, flows_cfg: Dict[str, Any]) -> List[str]:
        enabled = [
            k
            for k, v in (flows_cfg or {}).items()
            if isinstance(v, dict) and v.get("enabled")
        ]
        return [f"flow:{name}" for name in enabled]
