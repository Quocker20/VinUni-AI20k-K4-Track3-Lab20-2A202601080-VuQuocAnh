"""Supervisor / router implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self.settings = get_settings()

    def decide_route(self, state: ResearchState) -> str:
        """Evaluate shared state and return the next agent route or 'done'."""
        if state.iteration >= self.settings.max_iterations:
            logger.info(
                "Max iterations reached (%d/%d). Halting workflow.",
                state.iteration,
                self.settings.max_iterations,
            )
            return "done"

        if not state.sources or not state.research_notes:
            return "researcher"

        if not state.analysis_notes:
            return "analyst"

        if not state.final_answer:
            return "writer"

        return "done"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, record next route, and update trace."""
        next_route = self.decide_route(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_route",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
        )
        return state
