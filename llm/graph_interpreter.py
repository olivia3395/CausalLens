"""
llm/graph_interpreter.py
-------------------------
Uses the Anthropic Claude API to produce a plain-English interpretation
of the causal DAG: what the graph implies, key confounders, and potential
threats to validity.
"""

from __future__ import annotations

import os
from typing import Optional

import networkx as nx

import anthropic
from dotenv import load_dotenv

load_dotenv()


def interpret_causal_graph(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    dataset_description: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Ask Claude to interpret the causal DAG in plain English.

    Parameters
    ----------
    G                   : the causal DAG as a NetworkX DiGraph
    treatment           : treatment node name
    outcome             : outcome node name
    dataset_description : short description of the dataset / domain context
    api_key             : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

    Returns
    -------
    str  Plain-English interpretation from Claude.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return (
            "⚠️  No Anthropic API key found.  "
            "Set ANTHROPIC_API_KEY in your .env file to enable LLM interpretation."
        )

    # Summarise the graph for the prompt
    edges_str   = "\n".join(f"  {u} → {v}" for u, v in G.edges())
    confounders = [n for n in G.nodes() if n not in (treatment, outcome)]

    prompt = f"""You are a senior data scientist specialising in causal inference.

Dataset context:
{dataset_description}

Treatment variable : {treatment}
Outcome variable   : {outcome}
Confounders        : {', '.join(confounders)}

Causal DAG edges (directed):
{edges_str}

Please provide:
1. A concise (2–3 sentence) plain-English description of what this causal graph represents.
2. The top 2–3 confounders most likely to bias a naive treatment–outcome comparison, and why.
3. Any potential threats to the validity of this DAG (e.g. unmeasured confounding, selection bias).
4. One actionable business question this causal structure could help answer.

Keep your response under 300 words.  Use clear, non-technical language where possible."""

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
