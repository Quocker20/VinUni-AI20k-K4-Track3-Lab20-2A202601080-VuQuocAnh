"""Benchmark engine comparing single-agent vs multi-agent workflows."""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import yaml  # type: ignore[import-untyped]

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate the citation coverage ratio (claims with citations / total claims)."""
    if not state.final_answer:
        return 0.0

    # Count bracketed citations such as [Source 1], [A01], [KB-1]
    citation_pattern = re.compile(r"\[(?:Source\s*\d+|[A-Z0-9_-]+)\]", re.IGNORECASE)
    paragraphs = [p.strip() for p in state.final_answer.split("\n\n") if len(p.strip()) > 30]

    if not paragraphs:
        return 0.0

    paragraphs_with_citations = sum(1 for p in paragraphs if citation_pattern.search(p))
    return min(1.0, paragraphs_with_citations / len(paragraphs))


def compute_quality_score(state: ResearchState) -> float:
    """Evaluate report quality based on rubric criteria (0.0 to 10.0 scale).

    Rubric dimensions:
    - Structural completeness (Summary, Architecture/Methodology, Trade-offs, References): 4.0 pts
    - Citation grounding: 3.0 pts
    - Depth and length adequacy (300-800 words): 2.0 pts
    - Intermediate agent artifacts present: 1.0 pts
    """
    if not state.final_answer:
        return 0.0

    score = 0.0
    text = state.final_answer.lower()

    # 1. Structural headers
    required_sections = ["summary", "architecture", "trade-off", "reference"]
    for sec in required_sections:
        if sec in text:
            score += 1.0

    # 2. Citation coverage
    cov = compute_citation_coverage(state)
    score += cov * 3.0

    # 3. Word count adequacy
    word_count = len(state.final_answer.split())
    if 300 <= word_count <= 900:
        score += 2.0
    elif word_count > 150:
        score += 1.0

    # 4. Intermediate state fullness
    if state.research_notes and state.analysis_notes:
        score += 1.0
    elif state.research_notes:
        score += 0.5

    return round(min(10.0, max(0.0, score)), 1)


def compute_total_cost(state: ResearchState) -> float:
    """Sum total estimated cost from all agent steps."""
    total = 0.0
    for res in state.agent_results:
        cost = res.metadata.get("cost_usd")
        if cost is not None:
            total += float(cost)
    return total


def run_single_agent(query: str) -> ResearchState:
    """Execute single-agent baseline."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm_client = LLMClient()

    system_prompt = (
        "You are an expert AI research assistant. Conduct comprehensive research and write a "
        "well-structured, objective, and detailed technical summary addressing the user's request. "
        "Include sections for Executive Summary, Technical Details, Trade-offs, and References."
    )

    with trace_span("single_agent_baseline", {"query": query}):
        response = llm_client.complete(system_prompt=system_prompt, user_prompt=query)

    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    return state


def run_multi_agent(query: str) -> ResearchState:
    """Execute multi-agent LangGraph workflow."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    with trace_span("multi_agent_workflow", {"query": query}):
        result = workflow.run(state)
    return result


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute benchmark run and compute latency, cost, quality, citation coverage."""
    started = perf_counter()
    failure_rate = 0.0

    try:
        state = runner(query)
    except Exception as exc:
        logger.error("Benchmark runner '%s' failed: %s", run_name, exc)
        latency = perf_counter() - started
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"Error: {exc}",
        )
        return ResearchState(request=ResearchQuery(query=query), errors=[str(exc)]), metrics

    latency = perf_counter() - started
    cost = compute_total_cost(state)
    quality = compute_quality_score(state)
    citation_cov = compute_citation_coverage(state)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality,
        citation_coverage=citation_cov,
        failure_rate=failure_rate,
        notes=f"{len(state.sources)} sources, {state.iteration} turns",
    )
    return state, metrics


def run_full_benchmark_suite(queries: list[str] | None = None) -> list[BenchmarkMetrics]:
    """Run full benchmark comparing Single-Agent vs Multi-Agent across benchmark queries."""
    if queries is None:
        config_path = Path(__file__).resolve().parents[3] / "configs" / "lab_default.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            queries = data.get("benchmark", {}).get("queries", [])
        if not queries:
            queries = [
                "Research GraphRAG state-of-the-art and write a 500-word summary",
                "Compare single-agent and multi-agent workflows for customer support",
                "Summarize production guardrails for LLM agents",
            ]

    all_metrics: list[BenchmarkMetrics] = []

    for i, q in enumerate(queries, 1):
        logger.info("Running Benchmark Query %d/%d: %s", i, len(queries), q)

        # 1. Single Agent Baseline
        _, base_metric = run_benchmark(f"Single-Agent (Q{i})", q, run_single_agent)
        all_metrics.append(base_metric)

        # 2. Multi-Agent System
        _, multi_metric = run_benchmark(f"Multi-Agent (Q{i})", q, run_multi_agent)
        all_metrics.append(multi_metric)

    # Render and save markdown report
    report_content = render_markdown_report(all_metrics)
    report_dir = Path(__file__).resolve().parents[3] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "benchmark_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Benchmark report saved to %s", report_path)

    return all_metrics


if __name__ == "__main__":
    run_full_benchmark_suite()
