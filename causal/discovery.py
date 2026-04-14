"""
causal/discovery.py
-------------------
Build and visualise a causal DAG using DoWhy's graph utilities.

We use domain knowledge to hard-code sensible DAGs for each dataset
rather than running a purely data-driven discovery algorithm (which
tends to be unreliable on small tabular datasets). This mirrors how
practitioners actually use causal inference in industry.
"""

from __future__ import annotations

from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd


# ── Predefined DAGs ──────────────────────────────────────────────────────────

def _ibm_edges() -> list[tuple[str, str]]:
    """Domain-knowledge DAG for the IBM HR Attrition dataset."""
    return [
        # background → income
        ("Age",                       "HighIncome"),
        ("JobLevel",                  "HighIncome"),
        ("TotalWorkingYears",         "HighIncome"),
        ("YearsAtCompany",            "HighIncome"),
        # background → attrition
        ("Age",                       "Attrition_num"),
        ("JobSatisfaction",           "Attrition_num"),
        ("EnvironmentSatisfaction",   "Attrition_num"),
        ("WorkLifeBalance",           "Attrition_num"),
        ("OverTime_num",              "Attrition_num"),
        ("MaritalStatus_num",         "Attrition_num"),
        # income → attrition  (causal edge of interest)
        ("HighIncome",                "Attrition_num"),
        # job level mediates income and attrition
        ("JobLevel",                  "Attrition_num"),
        ("YearsSinceLastPromotion",   "Attrition_num"),
    ]


def _lalonde_edges() -> list[tuple[str, str]]:
    """Domain-knowledge DAG for the Lalonde NSW dataset."""
    return [
        # background → treatment assignment
        ("age",       "treat"),
        ("educ",      "treat"),
        ("black",     "treat"),
        ("hisp",      "treat"),
        ("married",   "treat"),
        ("nodegree",  "treat"),
        ("re74",      "treat"),
        ("re75",      "treat"),
        # background → outcome
        ("age",       "re78"),
        ("educ",      "re78"),
        ("black",     "re78"),
        ("hisp",      "re78"),
        ("married",   "re78"),
        ("nodegree",  "re78"),
        ("re74",      "re78"),
        ("re75",      "re78"),
        # treatment → outcome  (causal edge of interest)
        ("treat",     "re78"),
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def build_causal_graph(
    dataset: str,
    treatment: str,
    outcome: str,
) -> nx.DiGraph:
    """
    Return a NetworkX DiGraph for *dataset* ('ibm' or 'lalonde').

    Parameters
    ----------
    dataset   : 'ibm' or 'lalonde'
    treatment : treatment column name (used to validate edge presence)
    outcome   : outcome column name   (used to validate edge presence)
    """
    if dataset.lower() == "ibm":
        edges = _ibm_edges()
    elif dataset.lower() == "lalonde":
        edges = _lalonde_edges()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'ibm' or 'lalonde'.")

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # sanity check
    if not G.has_edge(treatment, outcome):
        raise ValueError(
            f"The DAG does not contain a direct edge {treatment} → {outcome}. "
            "Please check treatment / outcome names."
        )

    return G


def plot_causal_graph(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    title: str = "Causal DAG",
    figsize: tuple[int, int] = (12, 7),
) -> plt.Figure:
    """
    Draw the causal DAG with colour-coded nodes:
        🟢 green  = treatment
        🔴 red    = outcome
        🔵 blue   = confounder
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    # layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.5)

    # node colours
    node_colors = []
    for node in G.nodes():
        if node == treatment:
            node_colors.append("#2ecc71")   # green
        elif node == outcome:
            node_colors.append("#e74c3c")   # red
        else:
            node_colors.append("#3498db")   # blue

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=1800, alpha=0.92,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8, font_color="white", font_weight="bold",
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#555555", arrows=True,
        arrowstyle="-|>", arrowsize=18,
        width=1.6, connectionstyle="arc3,rad=0.05",
    )

    # legend
    legend_handles = [
        mpatches.Patch(color="#2ecc71", label=f"Treatment: {treatment}"),
        mpatches.Patch(color="#e74c3c", label=f"Outcome: {outcome}"),
        mpatches.Patch(color="#3498db", label="Confounder"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.axis("off")
    plt.tight_layout()
    return fig
