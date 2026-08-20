"""Optional critic agent implementation for fact checking and validation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer, check citation grounding, and append verification report."""
        logger.info("CriticAgent auditing report quality and citation grounding.")

        sources_summary = "\n".join(
            f"- [Source {i+1}]: {src.title}\n  Snippet: {src.snippet[:200]}"
            for i, src in enumerate(state.sources)
        )

        system_prompt = (
            "You are a Verification and Quality Assurance Critic. Evaluate the draft final report "
            "against the provided sources. Assess:\n"
            "1. Citation Grounding (Are cited claims genuinely found in the source snippets?)\n"
            "2. Hallucination Risk (Are there unsupported claims?)\n"
            "3. Quality Score (0-10) with brief justification."
        )
        user_prompt = (
            f"Sources:\n{sources_summary}\n\n"
            f"Draft Final Answer:\n{state.final_answer or 'No answer provided.'}\n\n"
            "Perform a concise audit of this report."
        )

        response = self.llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "critic_complete",
            {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
