"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes with citations."""

    name = "writer"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with grounded citations."""
        logger.info("WriterAgent synthesizing final research report.")

        sources_manifest = "\n".join(
            f"- [Source {i+1}]: {src.title} ({src.url or 'local'})"
            for i, src in enumerate(state.sources)
        )

        system_prompt = (
            "You are a Principal Technical Writer and Research Synthesizer. "
            "Your task is to write a comprehensive, professional ~500-word research summary "
            "based strictly on the research notes, analysis, and source documents provided.\n\n"
            "Requirements:\n"
            "1. Clear hierarchical structure (Executive Summary, Technical Architecture, "
            "Key Findings & Trade-offs, Future Outlook).\n"
            "2. MANDATORY: Every major technical claim must have inline bracketed citations "
            "referencing the source (e.g., [Source 1], [Source 2]).\n"
            "3. Conclude with a 'References' section listing each cited source."
        )

        user_prompt = (
            f"Topic / Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Sources Available:\n{sources_manifest}\n\n"
            f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or 'N/A'}\n\n"
            "Write the final technical report now with precise citations."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
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
        state.add_trace_event(
            "writer_complete",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
