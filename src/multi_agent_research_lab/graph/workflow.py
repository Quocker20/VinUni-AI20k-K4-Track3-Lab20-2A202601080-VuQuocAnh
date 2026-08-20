"""LangGraph multi-agent workflow implementation."""

from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph with LangGraph orchestration."""

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self._compiled_graph: Any | None = None

    def build(self) -> Any:
        """Create and compile the LangGraph workflow."""
        if self._compiled_graph is not None:
            return self._compiled_graph

        graph = StateGraph(ResearchState)

        # Register agent nodes
        graph.add_node("supervisor", lambda state: self.supervisor.run(state))
        graph.add_node("researcher", lambda state: self.researcher.run(state))
        graph.add_node("analyst", lambda state: self.analyst.run(state))
        graph.add_node("writer", lambda state: self.writer.run(state))

        # Entry point is always Supervisor
        graph.set_entry_point("supervisor")

        # Define conditional routing from supervisor
        def route_condition(state: ResearchState) -> str:
            if not state.route_history:
                return END
            last_route = state.route_history[-1]
            if last_route in {"researcher", "analyst", "writer"}:
                return last_route
            return END

        graph.add_conditional_edges(
            "supervisor",
            route_condition,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )

        # Worker nodes hand back control to supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        self._compiled_graph = graph.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return final state."""
        app = self.build()
        result: Any = app.invoke(state)
        if isinstance(result, ResearchState):
            return result
        elif isinstance(result, dict):
            return ResearchState.model_validate(result)
        return state
