"""Command-line entrypoint for the lab starter."""

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline research completion."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    llm_client = LLMClient()
    system_prompt = (
        "You are an expert AI research assistant. Conduct comprehensive research and write a "
        "well-structured, objective, and detailed technical summary addressing the user's request."
    )

    started = perf_counter()
    response = llm_client.complete(system_prompt=system_prompt, user_prompt=request.query)
    latency = perf_counter() - started

    state.final_answer = response.content
    state.add_trace_event(
        "baseline_execution",
        {
            "latency_seconds": latency,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )

    cost_str = f"${response.cost_usd:.5f}" if response.cost_usd is not None else "N/A"
    in_tok = response.input_tokens if response.input_tokens is not None else 0
    out_tok = response.output_tokens if response.output_tokens is not None else 0

    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline Response"))
    console.print(
        Panel.fit(
            f"[bold green]Latency:[/bold green] {latency:.3f}s | "
            f"[bold blue]Tokens:[/bold blue] {in_tok} in / {out_tok} out | "
            f"[bold yellow]Est. Cost:[/bold yellow] {cost_str}",
            title="Baseline Performance Metrics",
            style="cyan",
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()

    started = perf_counter()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    latency = perf_counter() - started

    # Calculate aggregate tokens and cost across all agent steps
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    for agent_res in result.agent_results:
        meta = agent_res.metadata
        total_input_tokens += meta.get("input_tokens") or 0
        total_output_tokens += meta.get("output_tokens") or 0
        total_cost_usd += meta.get("cost_usd") or 0.0

    if result.final_answer:
        console.print(Panel.fit(result.final_answer, title="Multi-Agent Final Report"))

    routes_str = " -> ".join(result.route_history)
    cost_str = f"${total_cost_usd:.5f}"

    console.print(
        Panel.fit(
            f"[bold green]Latency:[/bold green] {latency:.3f}s | "
            f"[bold blue]Tokens:[/bold blue] {total_input_tokens} in / {total_output_tokens} out | "
            f"[bold yellow]Est. Cost:[/bold yellow] {cost_str}\n"
            f"[bold magenta]Route History ({result.iteration} turns):[/bold magenta] {routes_str}\n"
            f"[bold cyan]Sources Retrieved:[/bold cyan] {len(result.sources)} documents",
            title="Multi-Agent Workflow Execution Summary",
            style="green",
        )
    )


if __name__ == "__main__":
    app()
