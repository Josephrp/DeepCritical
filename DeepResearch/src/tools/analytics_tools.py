"""
Analytics tools for DeepCritical using Pydantic AI patterns.

This module provides Pydantic AI tool wrappers for the analytics.py functionality,
integrating with the existing tool registry and datatypes.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from .base import ToolSpec, ToolRunner, ExecutionResult, registry
from ..utils.analytics import (
    record_request,
    last_n_days_df,
    last_n_days_avg_time_df,
)


class AnalyticsRequest(BaseModel):
    """Request model for analytics operations."""

    duration: Optional[float] = Field(None, description="Request duration in seconds")
    num_results: Optional[int] = Field(None, description="Number of results processed")

    class Config:
        json_schema_extra = {"example": {"duration": 2.5, "num_results": 4}}


class AnalyticsResponse(BaseModel):
    """Response model for analytics operations."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Operation result message")
    error: Optional[str] = Field(None, description="Error message if operation failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Request recorded successfully",
                "error": None,
            }
        }


class AnalyticsDataRequest(BaseModel):
    """Request model for analytics data retrieval."""

    days: int = Field(30, description="Number of days to retrieve data for")

    class Config:
        json_schema_extra = {"example": {"days": 30}}


class AnalyticsDataResponse(BaseModel):
    """Response model for analytics data retrieval."""

    data: List[Dict[str, Any]] = Field(..., description="Analytics data")
    success: bool = Field(..., description="Whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if operation failed")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {"date": "Jan 15", "count": 25, "full_date": "2024-01-15"},
                    {"date": "Jan 16", "count": 30, "full_date": "2024-01-16"},
                ],
                "success": True,
                "error": None,
            }
        }


class RecordRequestTool(ToolRunner):
    """Tool runner for recording request analytics."""

    def __init__(self):
        spec = ToolSpec(
            name="record_request",
            description="Record a request for analytics tracking",
            inputs={"duration": "FLOAT", "num_results": "INTEGER"},
            outputs={"success": "BOOLEAN", "message": "TEXT", "error": "TEXT"},
        )
        super().__init__(spec)

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        """Execute request recording operation."""
        try:
            import asyncio

            duration = params.get("duration")
            num_results = params.get("num_results")

            # Run async record_request
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(record_request(duration, num_results))
            finally:
                loop.close()

            return ExecutionResult(
                success=True,
                data={
                    "success": True,
                    "message": "Request recorded successfully",
                    "error": None,
                },
            )

        except Exception as e:
            return ExecutionResult(
                success=False, error=f"Failed to record request: {str(e)}"
            )


class GetAnalyticsDataTool(ToolRunner):
    """Tool runner for retrieving analytics data."""

    def __init__(self):
        spec = ToolSpec(
            name="get_analytics_data",
            description="Get analytics data for the specified number of days",
            inputs={"days": "INTEGER"},
            outputs={"data": "JSON", "success": "BOOLEAN", "error": "TEXT"},
        )
        super().__init__(spec)

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        """Execute analytics data retrieval operation."""
        try:
            days = params.get("days", 30)

            # Get analytics data
            df = last_n_days_df(days)
            data = df.to_dict("records")

            return ExecutionResult(
                success=True, data={"data": data, "success": True, "error": None}
            )

        except Exception as e:
            return ExecutionResult(
                success=False, error=f"Failed to get analytics data: {str(e)}"
            )


class GetAnalyticsTimeDataTool(ToolRunner):
    """Tool runner for retrieving analytics time data."""

    def __init__(self):
        spec = ToolSpec(
            name="get_analytics_time_data",
            description="Get analytics time data for the specified number of days",
            inputs={"days": "INTEGER"},
            outputs={"data": "JSON", "success": "BOOLEAN", "error": "TEXT"},
        )
        super().__init__(spec)

    def run(self, params: Dict[str, Any]) -> ExecutionResult:
        """Execute analytics time data retrieval operation."""
        try:
            days = params.get("days", 30)

            # Get analytics time data
            df = last_n_days_avg_time_df(days)
            data = df.to_dict("records")

            return ExecutionResult(
                success=True, data={"data": data, "success": True, "error": None}
            )

        except Exception as e:
            return ExecutionResult(
                success=False, error=f"Failed to get analytics time data: {str(e)}"
            )


# Pydantic AI Tool Functions
def record_request_tool(ctx: RunContext[Any]) -> str:
    """
    Record a request for analytics tracking.

    This tool records request metrics including duration and number of results
    for analytics and monitoring purposes.

    Args:
        duration: Request duration in seconds (optional)
        num_results: Number of results processed (optional)

    Returns:
        Success message or error description
    """
    # Extract parameters from context
    params = ctx.deps if isinstance(ctx.deps, dict) else {}

    # Create and run tool
    tool = RecordRequestTool()
    result = tool.run(params)

    if result.success:
        return result.data.get("message", "Request recorded successfully")
    else:
        return f"Failed to record request: {result.error}"


def get_analytics_data_tool(ctx: RunContext[Any]) -> str:
    """
    Get analytics data for the specified number of days.

    This tool retrieves request count analytics data for monitoring
    and reporting purposes.

    Args:
        days: Number of days to retrieve data for (optional, default: 30)

    Returns:
        JSON string containing analytics data
    """
    # Extract parameters from context
    params = ctx.deps if isinstance(ctx.deps, dict) else {}

    # Create and run tool
    tool = GetAnalyticsDataTool()
    result = tool.run(params)

    if result.success:
        return json.dumps(result.data.get("data", []))
    else:
        return f"Failed to get analytics data: {result.error}"


def get_analytics_time_data_tool(ctx: RunContext[Any]) -> str:
    """
    Get analytics time data for the specified number of days.

    This tool retrieves average request time analytics data for performance
    monitoring and optimization purposes.

    Args:
        days: Number of days to retrieve data for (optional, default: 30)

    Returns:
        JSON string containing analytics time data
    """
    # Extract parameters from context
    params = ctx.deps if isinstance(ctx.deps, dict) else {}

    # Create and run tool
    tool = GetAnalyticsTimeDataTool()
    result = tool.run(params)

    if result.success:
        return json.dumps(result.data.get("data", []))
    else:
        return f"Failed to get analytics time data: {result.error}"


@dataclass
class AnalyticsTool(ToolRunner):
    """Tool for analytics operations and metrics tracking."""

    def __init__(self):
        super().__init__(
            ToolSpec(
                name="analytics",
                description="Perform analytics operations and retrieve metrics",
                inputs={"operation": "TEXT", "days": "NUMBER", "parameters": "TEXT"},
                outputs={"result": "TEXT", "data": "TEXT"},
            )
        )

    def run(self, params: Dict[str, str]) -> ExecutionResult:
        operation = params.get("operation", "")
        days = int(params.get("days", "7"))

        if operation == "request_rate":
            # Calculate request rate using existing analytics functions
            df = last_n_days_df(days)
            rate = df["request_count"].sum() / days if not df.empty else 0.0
            return ExecutionResult(
                success=True,
                data={
                    "result": f"Average requests per day: {rate:.2f}",
                    "data": f"Rate: {rate}",
                },
                metrics={"days": days, "rate": rate},
            )
        elif operation == "response_time":
            # Calculate average response time
            df = last_n_days_avg_time_df(days)
            avg_time = df["avg_time"].mean() if not df.empty else 0.0
            return ExecutionResult(
                success=True,
                data={
                    "result": f"Average response time: {avg_time:.2f}s",
                    "data": f"Avg time: {avg_time}",
                },
                metrics={"days": days, "avg_time": avg_time},
            )
        else:
            return ExecutionResult(
                success=False, error=f"Unknown analytics operation: {operation}"
            )


# Register tools with the global registry
def register_analytics_tools():
    """Register analytics tools with the global registry."""
    registry.register("record_request", RecordRequestTool)
    registry.register("get_analytics_data", GetAnalyticsDataTool)
    registry.register("get_analytics_time_data", GetAnalyticsTimeDataTool)


# Auto-register when module is imported
register_analytics_tools()
registry.register("analytics", AnalyticsTool)
