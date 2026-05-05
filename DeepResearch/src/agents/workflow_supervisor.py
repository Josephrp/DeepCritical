from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from DeepResearch.src.datatypes.workflow_orchestration import (
    OrchestrationState,
    WorkflowExecution,
    WorkflowResult,
    WorkflowStatus,
)


@dataclass
class WorkflowSupervisor:
    """Run workflows with lifecycle tracking and robust failure capture."""

    state: OrchestrationState

    async def run_workflow(
        self,
        execution: WorkflowExecution,
        workflow_func: Callable[[dict[str, Any], dict[str, Any]], Any],
        *,
        timeout_s: float | None = None,
    ) -> WorkflowResult:
        start = time.time()
        execution.status = WorkflowStatus.RUNNING
        execution.start_time = datetime.now()

        try:
            coro = workflow_func(
                execution.input_data, execution.workflow_config.parameters
            )
            if timeout_s:
                result = await asyncio.wait_for(coro, timeout=timeout_s)
            else:
                result = await coro

            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            execution.output_data = result

            return WorkflowResult(
                execution_id=execution.execution_id,
                workflow_name=execution.workflow_config.name,
                status=WorkflowStatus.COMPLETED,
                output_data=result,
                execution_time=time.time() - start,
            )
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.now()
            execution.error_message = str(e)

            return WorkflowResult(
                execution_id=execution.execution_id,
                workflow_name=execution.workflow_config.name,
                status=WorkflowStatus.FAILED,
                output_data={},
                execution_time=time.time() - start,
                error_details={"error": str(e)},
            )
