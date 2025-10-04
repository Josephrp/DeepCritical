"""
Search Agent using Pydantic AI with integrated websearch and analytics tools.

This agent demonstrates how to use the websearch and analytics tools with Pydantic AI
for intelligent search and retrieval operations.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..tools.websearch_tools import web_search_tool, chunked_search_tool
from ..tools.analytics_tools import record_request_tool, get_analytics_data_tool
from ..tools.integrated_search_tools import integrated_search_tool, rag_search_tool


class SearchAgentConfig(BaseModel):
    """Configuration for the search agent."""

    model: str = Field("gpt-4", description="Model to use for the agent")
    enable_analytics: bool = Field(
        True, description="Whether to enable analytics tracking"
    )
    default_search_type: str = Field("search", description="Default search type")
    default_num_results: int = Field(4, description="Default number of results")
    chunk_size: int = Field(1000, description="Default chunk size")
    chunk_overlap: int = Field(0, description="Default chunk overlap")


class SearchQuery(BaseModel):
    """Search query model."""

    query: str = Field(..., description="The search query")
    search_type: Optional[str] = Field(
        None, description="Type of search: 'search' or 'news'"
    )
    num_results: Optional[int] = Field(None, description="Number of results to fetch")
    use_rag: bool = Field(False, description="Whether to use RAG-optimized search")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "artificial intelligence developments 2024",
                "search_type": "news",
                "num_results": 5,
                "use_rag": True,
            }
        }


class SearchResult(BaseModel):
    """Search result model."""

    query: str = Field(..., description="Original query")
    content: str = Field(..., description="Search results content")
    success: bool = Field(..., description="Whether the search was successful")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
    analytics_recorded: bool = Field(
        False, description="Whether analytics were recorded"
    )
    error: Optional[str] = Field(None, description="Error message if search failed")


class SearchAgent:
    """Search agent using Pydantic AI with integrated tools."""

    def __init__(self, config: SearchAgentConfig):
        self.config = config
        self.agent = Agent(
            model=config.model,
            system_prompt=self._get_system_prompt(),
            tools=[
                web_search_tool,
                chunked_search_tool,
                integrated_search_tool,
                rag_search_tool,
                record_request_tool,
                get_analytics_data_tool,
            ],
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the search agent."""
        return """You are an intelligent search agent that helps users find information on the web.

Your capabilities include:
1. Web search - Search for general information or news
2. Chunked search - Search and process results into chunks for analysis
3. Integrated search - Comprehensive search with analytics and RAG formatting
4. RAG search - Search optimized for retrieval-augmented generation
5. Analytics tracking - Record search metrics for monitoring

When performing searches:
- Use the most appropriate search tool for the user's needs
- For general information, use web_search_tool
- For analysis or RAG workflows, use integrated_search_tool or rag_search_tool
- Always provide clear, well-formatted results
- Include relevant metadata and sources when available

Be helpful, accurate, and provide comprehensive search results."""

    async def search(self, query: SearchQuery) -> SearchResult:
        """Perform a search using the agent."""
        try:
            # Prepare context for the agent
            context = {
                "query": query.query,
                "search_type": query.search_type or self.config.default_search_type,
                "num_results": query.num_results or self.config.default_num_results,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "use_rag": query.use_rag,
            }

            # Create the user message
            user_message = f"""Please search for: "{query.query}"

Search type: {context["search_type"]}
Number of results: {context["num_results"]}
Use RAG format: {query.use_rag}

Please provide comprehensive search results with proper formatting and source attribution."""

            # Run the agent
            result = await self.agent.run(user_message, deps=context)

            # Extract processing time if available
            processing_time = None
            analytics_recorded = False

            # Check if the result contains processing information
            if hasattr(result, "data") and isinstance(result.data, dict):
                processing_time = result.data.get("processing_time")
                analytics_recorded = result.data.get("analytics_recorded", False)

            return SearchResult(
                query=query.query,
                content=result.data if hasattr(result, "data") else str(result),
                success=True,
                processing_time=processing_time,
                analytics_recorded=analytics_recorded,
            )

        except Exception as e:
            return SearchResult(
                query=query.query, content="", success=False, error=str(e)
            )

    async def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for the specified number of days."""
        try:
            context = {"days": days}
            result = await self.agent.run(
                f"Get analytics data for the last {days} days", deps=context
            )
            return result.data if hasattr(result, "data") else {}
        except Exception as e:
            return {"error": str(e)}

    def create_rag_agent(self) -> Agent:
        """Create a specialized RAG agent for vector store integration."""
        return Agent(
            model=self.config.model,
            system_prompt="""You are a RAG (Retrieval-Augmented Generation) search specialist.

Your role is to:
1. Perform searches optimized for vector store integration
2. Convert search results into RAG-compatible formats
3. Ensure proper chunking and metadata for vector embeddings
4. Provide structured outputs for RAG workflows

Use rag_search_tool for all search operations to ensure compatibility with RAG systems.""",
            tools=[rag_search_tool, integrated_search_tool],
        )


# Example usage functions
async def example_basic_search():
    """Example of basic search functionality."""
    config = SearchAgentConfig(
        model="gpt-4",
        enable_analytics=True,
        default_search_type="search",
        default_num_results=3,
    )

    agent = SearchAgent(config)

    query = SearchQuery(
        query="artificial intelligence developments 2024",
        search_type="news",
        num_results=5,
    )

    result = await agent.search(query)
    print(f"Search successful: {result.success}")
    print(f"Content: {result.content[:200]}...")
    print(f"Analytics recorded: {result.analytics_recorded}")


async def example_rag_search():
    """Example of RAG-optimized search."""
    config = SearchAgentConfig(
        model="gpt-4", enable_analytics=True, chunk_size=1000, chunk_overlap=100
    )

    agent = SearchAgent(config)

    query = SearchQuery(
        query="machine learning algorithms", use_rag=True, num_results=3
    )

    result = await agent.search(query)
    print(f"RAG search successful: {result.success}")
    print(f"Processing time: {result.processing_time}s")


async def example_analytics():
    """Example of analytics retrieval."""
    config = SearchAgentConfig(enable_analytics=True)
    agent = SearchAgent(config)

    analytics = await agent.get_analytics(days=7)
    print(f"Analytics data: {analytics}")


if __name__ == "__main__":
    import asyncio

    # Run examples
    asyncio.run(example_basic_search())
    asyncio.run(example_rag_search())
    asyncio.run(example_analytics())
