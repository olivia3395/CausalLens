"""
llm/report_generator.py
------------------------
Generates a business-readable causal analysis report using Claude.

The report synthesises:
  - naive vs causal estimates (PSM, IPW, Double ML, Causal Forest)
  - refutation test results
  - actionable business recommendations
"""

from __future__ import annotations

import os
from typing import List, Optional

import anthropic
from dotenv import load_dotenv

from causal.effect_estimation import CausalResults
from causal.refutation import RefutationResult

load_dotenv()

_REPORT_SYSTEM = """You are a senior applied scientist writing a causal inference
analysis report for a business audience.  Be precise about statistics but
explain every number in plain English.  Structure your report with clear
section headers.  Do not use bullet lists — write in flowing paragraphs."""


def generate_full_report(
    results: CausalResults,
    refutations: List[RefutationResult],
    dataset_description: str,
    extra_context: str = "",
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a full business-ready causal analysis report via Claude.

    Parameters
    ----------
    results             : CausalResults from estimate_all_effects
    refutations         : list of RefutationResult from run_all_refutations
    dataset_description : short plain-English description of the dataset
    extra_context       : any additional context to include in the prompt
    api_key             : Anthropic API key (falls back to env var)

    Returns
    -------
    str  Formatted markdown report.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return (
            "⚠️  No Anthropic API key found.  "
            "Set ANTHROPIC_API_KEY in your .env file to enable report generation."
        )

    # ── Build estimates block ─────────────────────────────────────────────────
    estimates_block = f"""
Naive (unadjusted) ATE : {results.naive_ate}

Causal estimates (controlling for confounders):
  • PSM ATE            : {results.psm_ate   if results.psm_ate   is not None else 'N/A'}
  • IPW ATE            : {results.ipw_ate   if results.ipw_ate   is not None else 'N/A'}
  • Double ML ATE      : {results.dml_ate   if results.dml_ate   is not None else 'N/A'}
    95 % CI            : [{results.dml_ci_lower}, {results.dml_ate_upper}]
  • Causal Forest ATE  : {results.cfdml_ate if results.cfdml_ate is not None else 'N/A'}
    95 % CI            : [{results.cfdml_ci_lower}, {results.cfdml_ci_upper}]
""".strip()

    # ── Build refutation block ────────────────────────────────────────────────
    ref_lines = []
    for r in refutations:
        status = "PASS" if r.passed else "FAIL"
        ref_lines.append(
            f"  • {r.test_name}: new ATE = {r.new_ate}, "
            f"p-value = {r.p_value if r.p_value is not None else 'N/A'} → {status}"
        )
    refutation_block = "\n".join(ref_lines)

    prompt = f"""
Dataset: {dataset_description}
Treatment: {results.treatment}   |   Outcome: {results.outcome}
Sample: {results.n_total} observations  ({results.n_treated} treated, {results.n_control} control)
{extra_context}

=== CAUSAL ESTIMATES ===
{estimates_block}

=== REFUTATION TESTS ===
{refutation_block}

Please write a professional causal analysis report with the following sections:

## Executive Summary
One paragraph. State the causal question, the best estimate of the ATE,
and the confidence level based on refutation tests.

## Findings
Compare the naive estimate with the causal estimates. Explain in plain English
what the Double ML and Causal Forest ATEs mean (e.g. "receiving job training
increases 1978 earnings by $X on average"). Discuss consistency across methods.

## Robustness
Interpret the refutation test results. Do they support or undermine confidence
in the causal estimate?

## Business Implications
What should a business or policy-maker do with this finding? Suggest one
concrete, data-supported action.

## Caveats & Limitations
What key assumptions underlie this analysis? What additional data or experiments
(e.g. an RCT) would strengthen the conclusion?

Write in a clear, professional tone.  The audience is a senior product or
business leader who is not a statistician.
""".strip()

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=_REPORT_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
