from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..tools.base import registry, ExecutionResult


@dataclass
class ToolCaller:
    retries: int = 2

    def call(self, tool: str, params: Dict[str, Any]) -> ExecutionResult:
        runner = registry.make(tool)
        result = runner.run(params)
        if result.success:
            return result
        attempts = 0
        while attempts < self.retries and not result.success:
            result = runner.run(params)
            attempts += 1
        return result

    def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        bag: Dict[str, Any] = {}

        def materialize(p: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in p.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    key = v[2:-1]
                    out[k] = bag.get(key, "")
                else:
                    out[k] = v
            return out

        for step in plan:
            tool = step["tool"]
            params = materialize(step.get("params", {}))
            result = self.call(tool, params)
            if not result.success:
                break
            for k, v in result.data.items():
                bag[f"{tool}.{k}"] = v
                bag[k] = v
        return bag
