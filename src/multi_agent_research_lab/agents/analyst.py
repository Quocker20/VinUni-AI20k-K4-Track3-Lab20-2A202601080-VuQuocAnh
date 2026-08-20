"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("AnalystAgent analyzing research notes.")

        system_prompt = (
            "You are a Principal AI Systems Analyst. Analyze the provided research notes "
            "and source evidence. Structure your analysis into:\n"
            "1. Key Technical Claims & Validation (confirm claims supported by sources)\n"
            "2. Comparative Viewpoints & Trade-offs (e.g. latency, single vs multi-agent)\n"
            "3. Evidence Strength & Limitations (flag assumptions or weak evidence)\n"
            "4. Core Recommendations & Synthesis Framework"
        )
        user_prompt = (
            f"Original Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or 'No notes provided.'}\n\n"
            "Perform a rigorous analytical assessment of these findings."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        state.analysis_notes = response.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst_complete",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
