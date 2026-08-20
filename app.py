"""Streamlit Interactive Web Demo for Multi-Agent Research System.

VinUni AI20k - K4 - Track 3 - Lab 20
Orchestrated Technical Research with Supervisor, Researcher, Analyst, Writer & Critic Agents.
"""

import os
from time import perf_counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    compute_quality_score,
    compute_total_cost,
)
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Lab | VinUni AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #78909C;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #B0BEC5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .agent-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-supervisor { background-color: #5C6BC0; color: white; }
    .badge-researcher { background-color: #26A69A; color: white; }
    .badge-analyst { background-color: #FFA726; color: white; }
    .badge-writer { background-color: #AB47BC; color: white; }
    .badge-critic { background-color: #EC407A; color: white; }
    .source-box {
        background: rgba(33, 150, 243, 0.08);
        border-left: 4px solid #2196F3;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_single_agent_baseline(query: str, audience: str) -> tuple[ResearchState, float]:
    """Execute single-agent baseline."""
    request = ResearchQuery(query=query, audience=audience)
    state = ResearchState(request=request)
    llm = LLMClient()

    system_prompt = (
        "You are an expert AI research assistant. Conduct comprehensive research and write a "
        "well-structured, objective, and detailed technical summary addressing the user's request. "
        "Include sections for Executive Summary, Technical Details, Trade-offs, and References."
    )

    started = perf_counter()
    response = llm.complete(system_prompt=system_prompt, user_prompt=query)
    latency = perf_counter() - started

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
    state.add_trace_event("baseline_complete", {"latency": latency})
    return state, latency


def run_multi_agent_system(
    query: str, max_sources: int, audience: str
) -> tuple[ResearchState, float]:
    """Execute multi-agent LangGraph workflow."""
    request = ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    started = perf_counter()
    result = workflow.run(state)
    latency = perf_counter() - started
    return result, latency


# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=64)
    st.markdown("### ⚙️ System Settings")

    settings = get_settings()
    st.info(f"**Provider**: OpenRouter\n\n**Model**: `{settings.effective_model}`")

    mode = st.radio(
        "Select Operation Mode",
        [
            "🚀 Multi-Agent Workflow",
            "⚡ Single-Agent Baseline",
            "⚖️ Side-by-Side Comparison",
            "📊 Benchmark Analytics",
            "🗺️ Architecture & Graph",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎯 Quick Presets")
    presets = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
        "Evaluate agentic memory architectures (MemGPT vs Graph-based)",
    ]

    selected_preset = st.selectbox("Load sample topic:", ["-- Select a preset --"] + presets)

    st.markdown("---")
    max_sources = st.slider("Max Sources to Retrieve", min_value=1, max_value=10, value=5)
    audience = st.selectbox(
        "Target Audience",
        ["Technical Experts", "Software Engineers", "General AI Researchers"],
    )


# Main App Header
st.markdown('<div class="main-header">🧠 Multi-Agent Research System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "LangGraph Orchestrated Multi-Agent Architecture: "
    "Supervisor • Researcher • Analyst • Writer</div>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Mode 1: Multi-Agent Workflow
# -------------------------------------------------------------
if mode == "🚀 Multi-Agent Workflow":
    default_query = selected_preset if selected_preset != "-- Select a preset --" else presets[0]
    user_query = st.text_area(
        "Enter your research query / topic:",
        value=default_query,
        height=90,
        help="Specify the technical topic you want the multi-agent system to investigate.",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Run Multi-Agent", type="primary", use_container_width=True)

    if run_btn and user_query.strip():
        with st.status("🤖 Multi-Agent Workflow in progress...", expanded=True) as status:
            st.write("🎯 **Supervisor Agent**: Inspecting query & dispatching Researcher...")
            t_start = perf_counter()

            try:
                state, latency = run_multi_agent_system(user_query, max_sources, audience)
                status.update(
                    label="✅ Multi-Agent Workflow Completed Successfully!",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                status.update(label=f"❌ Error: {e}", state="error")
                st.error(f"Workflow execution failed: {e}")
                st.stop()

        # Aggregate metrics
        total_in_tokens = 0
        total_out_tokens = 0
        total_cost = 0.0
        for r in state.agent_results:
            meta = r.metadata
            total_in_tokens += meta.get("input_tokens") or 0
            total_out_tokens += meta.get("output_tokens") or 0
            total_cost += meta.get("cost_usd") or 0.0

        quality = compute_quality_score(state)
        citation_cov = compute_citation_coverage(state)

        # Top Metric Cards
        st.markdown("### 📈 Execution Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{latency:.2f}s</div>'
                f'<div class="metric-label">Latency</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{quality}/10</div>'
                f'<div class="metric-label">Quality Score</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{citation_cov:.0%}</div>'
                f'<div class="metric-label">Citation Coverage</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">${total_cost:.5f}</div>'
                f'<div class="metric-label">Est. Cost</div></div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{total_in_tokens + total_out_tokens}</div>'
                f'<div class="metric-label">Total Tokens</div></div>',
                unsafe_allow_html=True,
            )
        with m6:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{len(state.sources)}</div>'
                f'<div class="metric-label">Sources</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Route breadcrumb
        valid_roles = ["supervisor", "researcher", "analyst", "writer", "critic"]
        route_badges = " ➔ ".join(
            f'<span class="agent-badge badge-{r}">{r.upper()}</span>'
            if r in valid_roles
            else f'<span class="agent-badge badge-supervisor">{r.upper()}</span>'
            for r in state.route_history
        )
        st.markdown(
            f"**Execution Route ({state.iteration} turns)**: {route_badges}",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Tabs for deep inspection
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📄 Final Technical Report",
                "📚 Retrieved Sources",
                "📝 Researcher Notes",
                "🔍 Analyst Evaluation",
                "⏱️ Execution Trace & State",
            ]
        )

        with tab1:
            if state.final_answer:
                st.markdown(state.final_answer)
            else:
                st.warning("No final report generated.")

        with tab2:
            st.markdown(f"#### Retrieved Documents ({len(state.sources)})")
            for i, src in enumerate(state.sources, 1):
                with st.expander(f"📌 [Source {i}]: {src.title}", expanded=i <= 2):
                    if src.url:
                        st.markdown(f"**URL**: [{src.url}]({src.url})")
                    st.markdown(f"**Snippet**:\n> {src.snippet}")

        with tab3:
            st.markdown("#### Factual Notes Extracted by ResearcherAgent")
            if state.research_notes:
                st.markdown(state.research_notes)
            else:
                st.info("No research notes found.")

        with tab4:
            st.markdown("#### Structured Analysis by AnalystAgent")
            if state.analysis_notes:
                st.markdown(state.analysis_notes)
            else:
                st.info("No analysis notes found.")

        with tab5:
            st.markdown("#### State Trace Events")
            st.json(state.trace)
            st.markdown("#### Complete State Dump")
            st.json(state.model_dump())

# -------------------------------------------------------------
# Mode 2: Single-Agent Baseline
# -------------------------------------------------------------
elif mode == "⚡ Single-Agent Baseline":
    st.subheader("⚡ Single-Agent Baseline Evaluation")
    default_query = selected_preset if selected_preset != "-- Select a preset --" else presets[0]
    user_query = st.text_area(
        "Enter query for single-agent prompt:",
        value=default_query,
        height=90,
    )

    if st.button("⚡ Run Baseline", type="primary"):
        with st.spinner("Generating baseline response..."):
            state, latency = run_single_agent_baseline(user_query, audience)

        total_cost = compute_total_cost(state)
        quality = compute_quality_score(state)
        cov = compute_citation_coverage(state)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latency", f"{latency:.2f}s")
        m2.metric("Quality Score", f"{quality}/10")
        m3.metric("Citation Coverage", f"{cov:.0%}")
        m4.metric("Estimated Cost", f"${total_cost:.5f}")

        st.markdown("### Output Response")
        st.markdown(state.final_answer or "No answer.")

# -------------------------------------------------------------
# Mode 3: Side-by-Side Comparison
# -------------------------------------------------------------
elif mode == "⚖️ Side-by-Side Comparison":
    st.subheader("⚖️ Side-by-Side Comparison: Single-Agent vs Multi-Agent")
    default_query = selected_preset if selected_preset != "-- Select a preset --" else presets[0]
    user_query = st.text_area("Research Query for Comparison:", value=default_query, height=90)

    if st.button("⚖️ Run Both & Compare", type="primary"):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### ⚡ Single-Agent Baseline")
            with st.spinner("Running single-agent baseline..."):
                base_state, base_lat = run_single_agent_baseline(user_query, audience)
                base_cost = compute_total_cost(base_state)
                base_qual = compute_quality_score(base_state)
                base_cov = compute_citation_coverage(base_state)

            bm1, bm2, bm3, bm4 = st.columns(4)
            bm1.metric("Latency", f"{base_lat:.2f}s")
            bm2.metric("Quality", f"{base_qual}/10")
            bm3.metric("Citations", f"{base_cov:.0%}")
            bm4.metric("Cost", f"${base_cost:.5f}")

            with st.expander("View Single-Agent Response", expanded=True):
                st.markdown(base_state.final_answer or "")

        with c2:
            st.markdown("#### 🚀 Multi-Agent System")
            with st.spinner("Running multi-agent pipeline..."):
                multi_state, multi_lat = run_multi_agent_system(user_query, max_sources, audience)
                multi_cost = compute_total_cost(multi_state)
                multi_qual = compute_quality_score(multi_state)
                multi_cov = compute_citation_coverage(multi_state)

            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric(
                "Latency",
                f"{multi_lat:.2f}s",
                delta=f"{multi_lat - base_lat:.2f}s",
                delta_color="inverse",
            )
            mm2.metric("Quality", f"{multi_qual}/10", delta=f"{multi_qual - base_qual:+.1f}")
            mm3.metric("Citations", f"{multi_cov:.0%}", delta=f"{multi_cov - base_cov:+.0%}")
            mm4.metric(
                "Cost",
                f"${multi_cost:.5f}",
                delta=f"${multi_cost - base_cost:.5f}",
                delta_color="inverse",
            )

            with st.expander("View Multi-Agent Report", expanded=True):
                st.markdown(multi_state.final_answer or "")

# -------------------------------------------------------------
# Mode 4: Benchmark Analytics
# -------------------------------------------------------------
elif mode == "📊 Benchmark Analytics":
    st.subheader("📊 Empirical Benchmark Analytics")

    # Load benchmark report markdown
    report_file = os.path.join(os.path.dirname(__file__), "reports", "benchmark_report.md")
    if os.path.exists(report_file):
        with open(report_file, encoding="utf-8") as f:
            report_text = f.read()

        st.markdown(report_text)

        # Interactive chart data
        chart_data = pd.DataFrame(
            {
                "System": [
                    "Single-Agent (Q1)",
                    "Multi-Agent (Q1)",
                    "Single-Agent (Q2)",
                    "Multi-Agent (Q2)",
                    "Single-Agent (Q3)",
                    "Multi-Agent (Q3)",
                ],
                "Quality Score": [6.0, 8.8, 5.0, 8.9, 6.0, 9.1],
                "Citation Coverage (%)": [0, 60, 0, 64, 0, 69],
                "Latency (s)": [15.05, 30.77, 14.51, 31.34, 9.69, 30.19],
                "Cost ($ USD)": [0.00047, 0.00197, 0.00046, 0.00192, 0.00055, 0.00211],
            }
        )

        st.markdown("### 📈 Quality Score Comparison")
        st.bar_chart(chart_data.set_index("System")["Quality Score"])

        st.markdown("### 📚 Citation Coverage Comparison (%)")
        st.bar_chart(chart_data.set_index("System")["Citation Coverage (%)"])
    else:
        st.info("No benchmark report found. Run benchmark suite first.")

# -------------------------------------------------------------
# Mode 5: Architecture & Graph
# -------------------------------------------------------------
elif mode == "🗺️ Architecture & Graph":
    st.subheader("🗺️ System Architecture & LangGraph State Machine")

    st.markdown(
        """
        ```mermaid
        flowchart TD
            Start([Start]) --> Supervisor{Supervisor Agent}
            
            Supervisor -->|No sources or notes| Researcher[Researcher Agent]
            Supervisor -->|No analysis notes| Analyst[Analyst Agent]
            Supervisor -->|No final answer| Writer[Writer Agent]
            Supervisor -->|Report ready or Max Iterations| Done([END])
            
            Researcher --> Supervisor
            Analyst --> Supervisor
            Writer --> Supervisor
        ```
        """
    )

    st.markdown("### 🛡️ Production Guardrails")
    st.markdown(
        """
        1. **Max Iterations (Loop Breaker)**: `Settings.max_iterations = 6` limits runaway cycles.
        2. **Network Exponential Backoff**: `@retry` decorator on `LLMClient` via `tenacity`.
        3. **Offline Knowledge Corpus Fallback**: `SearchClient` fallback to 30 offline JSON topics.
        4. **Pydantic State Validation**: Strict typing across `ResearchState`, `SourceDocument`.
        """
    )

    st.markdown("### 👥 Agent Roles Matrix")
    roles_df = pd.DataFrame(
        [
            {
                "Role": "Supervisor",
                "Responsibility": "Inspects state & orchestrates next route",
                "Input": "ResearchState",
                "Output": "Route decision & history",
            },
            {
                "Role": "Researcher",
                "Responsibility": "Searches evidence & extracts fact notes",
                "Input": "Query & max sources",
                "Output": "sources & research_notes",
            },
            {
                "Role": "Analyst",
                "Responsibility": "Weighs claims & analyzes trade-offs",
                "Input": "research_notes & sources",
                "Output": "analysis_notes",
            },
            {
                "Role": "Writer",
                "Responsibility": "Synthesizes report with inline citations",
                "Input": "All notes & sources",
                "Output": "final_answer",
            },
            {
                "Role": "Critic",
                "Responsibility": "Audits grounding & hallucination risk",
                "Input": "sources & final_answer",
                "Output": "critic audit result",
            },
        ]
    )
    st.dataframe(roles_df, use_container_width=True, hide_index=True)
