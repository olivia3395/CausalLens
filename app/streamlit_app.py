"""
app/streamlit_app.py
---------------------
CausalLens — interactive Streamlit UI for causal analysis.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# make sure project root is on path when launched from app/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── project imports ───────────────────────────────────────────────────────────
from data.load_data import (
    load_ibm, load_lalonde, dataset_summary,
    IBM_TREATMENT, IBM_OUTCOME, IBM_CONFOUNDERS,
    LALONDE_TREATMENT, LALONDE_OUTCOME, LALONDE_CONFOUNDERS,
)
from causal.discovery import build_causal_graph, plot_causal_graph
from causal.effect_estimation import estimate_all_effects
from causal.refutation import run_all_refutations, refutations_to_df
from llm.graph_interpreter import interpret_causal_graph
from llm.report_generator import generate_full_report

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CausalLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 800; color: #1a1a2e; }
    .sub-title  { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        border-left: 4px solid #3a86ff;
    }
    .metric-val  { font-size: 1.6rem; font-weight: 700; color: #3a86ff; }
    .metric-label{ font-size: 0.82rem; color: #666; margin-top: 2px; }
    .section-header {
        font-size: 1.3rem; font-weight: 700;
        border-bottom: 2px solid #3a86ff;
        padding-bottom: 6px; margin-top: 1.5rem; margin-bottom: 0.8rem;
    }
    .pass-badge { color: #27ae60; font-weight: 700; }
    .fail-badge { color: #e74c3c; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/graph.png", width=60)
    st.markdown("## 🔍 CausalLens")
    st.markdown("*LLM-Augmented Causal Inference*")
    st.divider()

    dataset_choice = st.selectbox(
        "📂 Dataset",
        ["IBM HR Attrition", "Lalonde Job Training (NSW)"],
    )

    api_key = st.text_input(
        "🔑 Anthropic API Key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required for LLM interpretation and report generation.",
    )

    st.divider()
    n_refut_sims = st.slider("Refutation simulations", 50, 300, 100, step=50)
    run_button   = st.button("🚀 Run Full Analysis", use_container_width=True, type="primary")

    st.divider()
    st.caption("Built with DoWhy · EconML · Claude API")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 CausalLens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">LLM-augmented causal discovery · ATE estimation '
    '(PSM / IPW / Double ML / Causal Forest) · Refutation testing · '
    'AI-generated business reports</div>',
    unsafe_allow_html=True,
)

# ── Dataset config ────────────────────────────────────────────────────────────
if dataset_choice == "IBM HR Attrition":
    loader      = load_ibm
    treatment   = IBM_TREATMENT
    outcome     = IBM_OUTCOME
    confounders = IBM_CONFOUNDERS
    dataset_key = "ibm"
    dataset_desc = (
        "IBM HR Employee Attrition dataset (n≈1 470). "
        "We investigate whether earning above-median income (HighIncome) causally reduces "
        "the probability of employee attrition (leaving the company), "
        "after controlling for age, job level, satisfaction scores, overtime, and tenure."
    )
else:
    loader      = load_lalonde
    treatment   = LALONDE_TREATMENT
    outcome     = LALONDE_OUTCOME
    confounders = LALONDE_CONFOUNDERS
    dataset_key = "lalonde"
    dataset_desc = (
        "Lalonde NSW Job-Training dataset — a classic benchmark in causal inference. "
        "We estimate whether participating in a job-training programme (treat=1) "
        "causally increases 1978 earnings (re78), controlling for age, education, "
        "race, marital status, prior earnings (re74/re75), and degree status."
    )

# ── Load data (always visible) ────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    try:
        df = loader()
        summary = dataset_summary(df, treatment, outcome)
        data_loaded = True
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        data_loaded = False

if data_loaded:
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    def _metric(col, val, label):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _metric(c1, f"{summary['n_total']:,}", "Total Observations")
    _metric(c2, f"{summary['n_treated']:,}", "Treated")
    _metric(c3, f"{summary['n_control']:,}", "Control")
    _metric(c4, f"{summary['outcome_mean_treated']:.3f}", f"Mean {outcome} | Treated")
    _metric(c5, f"{summary['naive_ate']:+.3f}", "Naive ATE (unadjusted)")

    with st.expander("🔎 Preview raw data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

# ── Main analysis (triggered by button) ──────────────────────────────────────
if run_button and data_loaded:

    # ── Causal Graph ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🕸️ Causal DAG</div>', unsafe_allow_html=True)

    with st.spinner("Building causal graph..."):
        G   = build_causal_graph(dataset_key, treatment, outcome)
        fig = plot_causal_graph(G, treatment, outcome, title=f"Causal DAG — {dataset_choice}")

    col_graph, col_interp = st.columns([1.1, 0.9])
    with col_graph:
        st.pyplot(fig)
        plt.close(fig)

    with col_interp:
        st.markdown("**🤖 LLM Graph Interpretation**")
        if api_key:
            with st.spinner("Claude is reading the graph..."):
                interp = interpret_causal_graph(G, treatment, outcome, dataset_desc, api_key=api_key)
            st.markdown(interp)
        else:
            st.info("Add your Anthropic API key in the sidebar to enable LLM graph interpretation.")

    # ── Effect Estimation ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📐 Causal Effect Estimation</div>', unsafe_allow_html=True)

    with st.spinner("Running PSM · IPW · Double ML · Causal Forest  (this may take ~30 s)..."):
        results = estimate_all_effects(
            df, treatment, outcome, confounders,
            dataset_name=dataset_choice,
            graph=G,
        )

    if results.errors:
        with st.expander("⚠️ Estimation warnings"):
            for err in results.errors:
                st.warning(err)

    summary_df = results.summary_df()
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # visual comparison
    st.markdown("**Estimates at a glance**")
    plot_df = summary_df[summary_df["ATE"] != "—"].copy()
    plot_df["ATE"] = pd.to_numeric(plot_df["ATE"], errors="coerce")
    plot_df = plot_df.dropna(subset=["ATE"])

    if not plot_df.empty:
        fig2, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#3a86ff" if v >= 0 else "#e74c3c" for v in plot_df["ATE"]]
        bars = ax.barh(plot_df["Method"], plot_df["ATE"], color=colors, height=0.55)
        ax.axvline(0, color="#333", linewidth=1.2, linestyle="--")
        ax.axvline(summary["naive_ate"], color="#f39c12", linewidth=1.4,
                   linestyle=":", label=f"Naive ATE ({summary['naive_ate']:+.4f})")
        ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=9)
        ax.set_xlabel("Average Treatment Effect (ATE)")
        ax.set_title("Causal vs Naive ATE", fontweight="bold")
        ax.legend(fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Refutation Tests ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧪 Refutation Tests</div>', unsafe_allow_html=True)
    st.caption(
        "Refutation tests check whether the causal estimate is an artefact. "
        "✅ PASS = estimate is robust to that check."
    )

    ref_ate = results.dml_ate or results.psm_ate or results.naive_ate
    with st.spinner(f"Running {n_refut_sims} simulations per test..."):
        refutations = run_all_refutations(
            df, treatment, outcome, confounders,
            original_ate=ref_ate,
            n_simulations=n_refut_sims,
        )

    ref_df = refutations_to_df(refutations)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    for r in refutations:
        badge = "pass-badge" if r.passed else "fail-badge"
        st.markdown(
            f'<span class="{badge}">{r.test_name}</span>: {r.interpretation}',
            unsafe_allow_html=True,
        )

    # ── LLM Business Report ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📝 AI-Generated Business Report</div>', unsafe_allow_html=True)

    if api_key:
        with st.spinner("Claude is writing the report..."):
            report = generate_full_report(
                results=results,
                refutations=refutations,
                dataset_description=dataset_desc,
                api_key=api_key,
            )
        st.markdown(report)

        # download
        st.download_button(
            "⬇️ Download Report (.md)",
            data=report,
            file_name=f"causallens_report_{dataset_key}.md",
            mime="text/markdown",
        )
    else:
        st.info("Add your Anthropic API key in the sidebar to generate the business report.")

    # ── Results JSON export ───────────────────────────────────────────────────
    with st.expander("📦 Raw results (JSON)"):
        import json
        st.json(results.to_dict())

elif not data_loaded:
    st.warning("Could not load dataset.  Check your internet connection and try again.")
else:
    st.info("👈 Configure options in the sidebar and click **Run Full Analysis** to start.")
