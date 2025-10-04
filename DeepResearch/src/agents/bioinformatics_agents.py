"""
Bioinformatics agents for data fusion and reasoning tasks.

This module implements specialized agents using Pydantic AI for bioinformatics
data processing, fusion, and reasoning tasks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from ..datatypes.bioinformatics import (
    GOAnnotation,
    PubMedPaper,
    FusedDataset,
    ReasoningTask,
    DataFusionRequest,
)


class BioinformaticsAgentDeps(BaseModel):
    """Dependencies for bioinformatics agents."""

    config: Dict[str, Any] = Field(default_factory=dict)
    data_sources: List[str] = Field(default_factory=list)
    quality_threshold: float = Field(0.8, ge=0.0, le=1.0)

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "BioinformaticsAgentDeps":
        """Create dependencies from configuration."""
        bioinformatics_config = config.get("bioinformatics", {})
        quality_config = bioinformatics_config.get("quality", {})

        return cls(
            config=config,
            quality_threshold=quality_config.get("default_threshold", 0.8),
            **kwargs,
        )


class DataFusionResult(BaseModel):
    """Result of data fusion operation."""

    success: bool = Field(..., description="Whether fusion was successful")
    fused_dataset: Optional[FusedDataset] = Field(None, description="Fused dataset")
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages")
    processing_time: float = Field(0.0, description="Processing time in seconds")


class ReasoningResult(BaseModel):
    """Result of reasoning task."""

    success: bool = Field(..., description="Whether reasoning was successful")
    answer: str = Field(..., description="Reasoning answer")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Supporting evidence"
    )
    reasoning_chain: List[str] = Field(
        default_factory=list, description="Reasoning steps"
    )


class DataFusionAgent:
    """Agent for fusing bioinformatics data from multiple sources."""

    def __init__(
        self,
        model_name: str = "anthropic:claude-sonnet-4-0",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.config = config or {}
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[BioinformaticsAgentDeps, DataFusionResult]:
        """Create the data fusion agent."""
        # Get model from config or use default
        bioinformatics_config = self.config.get("bioinformatics", {})
        agents_config = bioinformatics_config.get("agents", {})
        data_fusion_config = agents_config.get("data_fusion", {})

        model_name = data_fusion_config.get("model", self.model_name)
        model = AnthropicModel(model_name)

        # Get system prompt from config or use default
        system_prompt = data_fusion_config.get(
            "system_prompt",
            """You are a bioinformatics data fusion specialist. Your role is to:
1. Analyze data fusion requests and identify relevant data sources
2. Apply quality filters and evidence code requirements
3. Create fused datasets that combine multiple bioinformatics sources
4. Ensure data consistency and cross-referencing
5. Generate quality metrics for the fused dataset

Focus on creating high-quality, scientifically sound fused datasets that can be used for reasoning tasks.
Always validate evidence codes and apply appropriate quality thresholds.""",
        )

        agent = Agent(
            model=model,
            deps_type=BioinformaticsAgentDeps,
            result_type=DataFusionResult,
            system_prompt=system_prompt,
        )

        return agent

    async def fuse_data(
        self, request: DataFusionRequest, deps: BioinformaticsAgentDeps
    ) -> DataFusionResult:
        """Fuse data from multiple sources based on the request."""

        fusion_prompt = f"""
        Fuse bioinformatics data according to the following request:
        
        Fusion Type: {request.fusion_type}
        Source Databases: {", ".join(request.source_databases)}
        Filters: {request.filters}
        Quality Threshold: {request.quality_threshold}
        Max Entities: {request.max_entities}
        
        Please create a fused dataset that:
        1. Combines data from the specified sources
        2. Applies the specified filters
        3. Maintains data quality above the threshold
        4. Includes proper cross-references between entities
        5. Generates appropriate quality metrics
        
        Return a DataFusionResult with the fused dataset and quality metrics.
        """

        result = await self.agent.run(fusion_prompt, deps=deps)
        return result.data


class GOAnnotationAgent:
    """Agent for processing GO annotations with PubMed context."""

    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        self.model_name = model_name
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[BioinformaticsAgentDeps, List[GOAnnotation]]:
        """Create the GO annotation agent."""
        model = AnthropicModel(self.model_name)

        agent = Agent(
            model=model,
            deps_type=BioinformaticsAgentDeps,
            result_type=List[GOAnnotation],
            system_prompt="""You are a GO annotation specialist. Your role is to:
1. Process GO annotations with PubMed paper context
2. Filter annotations based on evidence codes (prioritize IDA - gold standard)
3. Extract relevant information from paper abstracts and full text
4. Create high-quality annotations with proper cross-references
5. Ensure annotations meet quality standards

Focus on creating annotations that can be used for reasoning tasks, with emphasis on experimental evidence (IDA, EXP) over computational predictions.""",
        )

        return agent

    async def process_annotations(
        self,
        annotations: List[Dict[str, Any]],
        papers: List[PubMedPaper],
        deps: BioinformaticsAgentDeps,
    ) -> List[GOAnnotation]:
        """Process GO annotations with PubMed context."""

        processing_prompt = f"""
        Process the following GO annotations with PubMed paper context:
        
        Annotations: {len(annotations)} annotations
        Papers: {len(papers)} papers
        
        Please:
        1. Match annotations with their corresponding papers
        2. Filter for high-quality evidence codes (IDA, EXP preferred)
        3. Extract relevant context from paper abstracts
        4. Create properly structured GOAnnotation objects
        5. Ensure all required fields are populated
        
        Return a list of processed GOAnnotation objects.
        """

        result = await self.agent.run(processing_prompt, deps=deps)
        return result.data


class ReasoningAgent:
    """Agent for performing reasoning tasks on fused bioinformatics data."""

    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        self.model_name = model_name
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[BioinformaticsAgentDeps, ReasoningResult]:
        """Create the reasoning agent."""
        model = AnthropicModel(self.model_name)

        agent = Agent(
            model=model,
            deps_type=BioinformaticsAgentDeps,
            result_type=ReasoningResult,
            system_prompt="""You are a bioinformatics reasoning specialist. Your role is to:
1. Analyze reasoning tasks based on fused bioinformatics data
2. Apply multi-source evidence integration
3. Provide scientifically sound reasoning chains
4. Assess confidence levels based on evidence quality
5. Identify supporting evidence from multiple data sources

Focus on integrative reasoning that goes beyond reductionist approaches, considering:
- Gene co-occurrence patterns
- Protein-protein interactions
- Expression correlations
- Functional annotations
- Structural similarities
- Drug-target relationships

Always provide clear reasoning chains and confidence assessments.""",
        )

        return agent

    async def perform_reasoning(
        self, task: ReasoningTask, dataset: FusedDataset, deps: BioinformaticsAgentDeps
    ) -> ReasoningResult:
        """Perform reasoning task on fused dataset."""

        reasoning_prompt = f"""
        Perform the following reasoning task using the fused bioinformatics dataset:
        
        Task: {task.task_type}
        Question: {task.question}
        Difficulty: {task.difficulty_level}
        Required Evidence: {[code.value for code in task.required_evidence]}
        
        Dataset Information:
        - Total Entities: {dataset.total_entities}
        - Source Databases: {", ".join(dataset.source_databases)}
        - GO Annotations: {len(dataset.go_annotations)}
        - PubMed Papers: {len(dataset.pubmed_papers)}
        - Gene Expression Profiles: {len(dataset.gene_expression_profiles)}
        - Drug Targets: {len(dataset.drug_targets)}
        - Protein Structures: {len(dataset.protein_structures)}
        - Protein Interactions: {len(dataset.protein_interactions)}
        
        Please:
        1. Analyze the question using multi-source evidence
        2. Apply integrative reasoning (not just reductionist approaches)
        3. Consider cross-database relationships
        4. Provide a clear reasoning chain
        5. Assess confidence based on evidence quality
        6. Identify supporting evidence from multiple sources
        
        Return a ReasoningResult with your analysis.
        """

        result = await self.agent.run(reasoning_prompt, deps=deps)
        return result.data


class DataQualityAgent:
    """Agent for assessing data quality and consistency."""

    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        self.model_name = model_name
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent[BioinformaticsAgentDeps, Dict[str, float]]:
        """Create the data quality agent."""
        model = AnthropicModel(self.model_name)

        agent = Agent(
            model=model,
            deps_type=BioinformaticsAgentDeps,
            result_type=Dict[str, float],
            system_prompt="""You are a bioinformatics data quality specialist. Your role is to:
1. Assess data quality across multiple bioinformatics sources
2. Calculate consistency metrics between databases
3. Identify potential data conflicts or inconsistencies
4. Generate quality scores for fused datasets
5. Recommend quality improvements

Focus on:
- Evidence code distribution and quality
- Cross-database consistency
- Completeness of annotations
- Temporal consistency (recent vs. older data)
- Source reliability and curation standards""",
        )

        return agent

    async def assess_quality(
        self, dataset: FusedDataset, deps: BioinformaticsAgentDeps
    ) -> Dict[str, float]:
        """Assess quality of fused dataset."""

        quality_prompt = f"""
        Assess the quality of the following fused bioinformatics dataset:
        
        Dataset: {dataset.name}
        Source Databases: {", ".join(dataset.source_databases)}
        Total Entities: {dataset.total_entities}
        
        Component Counts:
        - GO Annotations: {len(dataset.go_annotations)}
        - PubMed Papers: {len(dataset.pubmed_papers)}
        - Gene Expression Profiles: {len(dataset.gene_expression_profiles)}
        - Drug Targets: {len(dataset.drug_targets)}
        - Protein Structures: {len(dataset.protein_structures)}
        - Protein Interactions: {len(dataset.protein_interactions)}
        
        Please calculate quality metrics including:
        1. Evidence code quality distribution
        2. Cross-database consistency
        3. Completeness scores
        4. Temporal relevance
        5. Source reliability
        6. Overall quality score
        
        Return a dictionary of quality metrics with scores between 0.0 and 1.0.
        """

        result = await self.agent.run(quality_prompt, deps=deps)
        return result.data


class BioinformaticsAgent:
    """Main bioinformatics agent that coordinates all bioinformatics operations."""

    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        self.model_name = model_name
        self.orchestrator = AgentOrchestrator(model_name)

    async def process_request(
        self, request: DataFusionRequest, deps: BioinformaticsAgentDeps
    ) -> tuple[FusedDataset, ReasoningResult, Dict[str, float]]:
        """Process a complete bioinformatics request end-to-end."""
        # Create reasoning dataset
        dataset, quality_metrics = await self.orchestrator.create_reasoning_dataset(
            request, deps
        )

        # Create a reasoning task for the request
        reasoning_task = ReasoningTask(
            task_id="main_task",
            task_type="integrative_analysis",
            question=request.reasoning_question or "Analyze the fused dataset",
            difficulty_level="moderate",
            required_evidence=[],  # Will use default evidence requirements
            timeout_seconds=300,
        )

        # Perform reasoning
        reasoning_result = await self.orchestrator.perform_integrative_reasoning(
            reasoning_task, dataset, deps
        )

        return dataset, reasoning_result, quality_metrics


class AgentOrchestrator:
    """Orchestrator for coordinating multiple bioinformatics agents."""

    def __init__(self, model_name: str = "anthropic:claude-sonnet-4-0"):
        self.model_name = model_name
        self.fusion_agent = DataFusionAgent(model_name)
        self.go_agent = GOAnnotationAgent(model_name)
        self.reasoning_agent = ReasoningAgent(model_name)
        self.quality_agent = DataQualityAgent(model_name)

    async def create_reasoning_dataset(
        self, request: DataFusionRequest, deps: BioinformaticsAgentDeps
    ) -> tuple[FusedDataset, Dict[str, float]]:
        """Create a reasoning dataset by fusing multiple data sources."""

        # Step 1: Fuse data from multiple sources
        fusion_result = await self.fusion_agent.fuse_data(request, deps)

        if not fusion_result.success:
            raise ValueError(f"Data fusion failed: {fusion_result.errors}")

        dataset = fusion_result.fused_dataset

        # Step 2: Assess data quality
        quality_metrics = await self.quality_agent.assess_quality(dataset, deps)

        # Update dataset with quality metrics
        dataset.quality_metrics = quality_metrics

        return dataset, quality_metrics

    async def perform_integrative_reasoning(
        self, task: ReasoningTask, dataset: FusedDataset, deps: BioinformaticsAgentDeps
    ) -> ReasoningResult:
        """Perform integrative reasoning using multiple data sources."""

        # Perform reasoning with multi-source evidence
        reasoning_result = await self.reasoning_agent.perform_reasoning(
            task, dataset, deps
        )

        return reasoning_result

    async def process_go_pubmed_fusion(
        self,
        go_annotations: List[Dict[str, Any]],
        pubmed_papers: List[PubMedPaper],
        deps: BioinformaticsAgentDeps,
    ) -> List[GOAnnotation]:
        """Process GO annotations with PubMed context for reasoning tasks."""

        # Process annotations with paper context
        processed_annotations = await self.go_agent.process_annotations(
            go_annotations, pubmed_papers, deps
        )

        return processed_annotations
