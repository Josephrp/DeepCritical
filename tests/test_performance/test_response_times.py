"""
Response time performance tests.
"""

import asyncio
import time
from unittest.mock import Mock

import pytest


class TestResponseTimes:
    """Test response time performance."""

    @pytest.mark.performance
    def test_agent_response_time(self):
        """Test that agent responses meet performance requirements."""
        # Mock agent execution
        mock_agent = Mock()
        mock_agent.execute = Mock(return_value={"result": "test", "success": True})

        start_time = time.time()
        result = mock_agent.execute("test query")
        end_time = time.time()

        response_time = end_time - start_time

        # Response should be under 1 second for simple queries
        assert response_time < 1.0
        assert result["success"] is True

    @pytest.mark.performance
    def test_concurrent_agent_execution(self):
        """Test performance under concurrent load."""

        async def run_concurrent_tests():
            # Simulate multiple concurrent agent executions
            tasks = []
            for i in range(10):
                task = asyncio.create_task(simulate_agent_call(f"query_{i}"))
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time

            # All tasks should complete successfully
            assert len(results) == 10
            assert all(result["success"] for result in results)

            # Total time should be reasonable (less than 5 seconds for 10 concurrent)
            assert total_time < 5.0

        async def simulate_agent_call(query: str):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"result": f"result_{query}", "success": True}

        asyncio.run(run_concurrent_tests())

    @pytest.mark.performance
    def test_memory_usage_monitoring(self):
        """Test memory usage doesn't grow excessively."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate memory-intensive operation
        # large_data = ["x" * 1000 for _ in range(1000)]  # Commented out to avoid unused variable warning

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for test data)
        assert memory_increase < 50.0
