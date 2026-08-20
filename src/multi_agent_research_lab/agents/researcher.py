"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.search_client = SearchClient()
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("ResearcherAgent gathering sources for query: %s", state.request.query)

        # Retrieve relevant source documents
        sources = self.search_client.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = sources

        sources_text = "\n\n".join(
            f"[Source {i+1}]: {src.title}\nURL/ID: {src.url or 'N/A'}\nSnippet: {src.snippet}"
            for i, src in enumerate(sources)
        )

        system_prompt = (
            "You are a Senior Research Specialist. Given the research query and retrieved sources, "
            "synthesize comprehensive, objective research notes. Extract key facts, architectural "
            "mechanisms, technical trade-offs, and empirical findings. Explicitly reference source "
            "numbers like [Source 1], [Source 2] where facts originate."
        )
        user_prompt = (
            f"Research Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Retrieved Evidence:\n{sources_text}\n\n"
            "Produce clear, bulleted technical research notes with cited evidence."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.research_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_complete",
            {
                "num_sources": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
