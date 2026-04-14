"""
causal/refutation.py
---------------------
Refutation tests validate whether a causal estimate is robust or a
statistical artefact.  We run three standard DoWhy refutations:

  1. Placebo Treatment       — replace treatment with random noise; ATE should → 0
  2. Random Common Cause     — add a spurious confounder; ATE should be stable
  3. Data Subset             — re-estimate on 80 % bootstrap; ATE should be stable

Results include the new estimated ATE, p-value, and a PASS/FAIL verdict.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

warnings.filterwarnings("ignore")


@dataclass
class RefutationResult:
    test_name: str
    original_ate: float
    new_ate: float
    p_value: Optional[float]
    passed: bool        # True = estimate is robust
    interpretation: str


def run_all_refutations(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    original_ate: float,
    n_simulations: int = 100,
) -> List[RefutationResult]:
    """
    Run three DoWhy refutation tests and return a list of RefutationResult.

    Parameters
    ----------
    df            : cleaned DataFrame
    treatment     : treatment column name
    outcome       : outcome column name
    confounders   : list of confounder column names
    original_ate  : the ATE we are trying to validate (e.g. from Double ML)
    n_simulations : number of bootstrap rounds for random_common_cause / subset
    """
    cols = [treatment, outcome] + confounders
    df = df[cols].dropna().copy()

    causal_model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
    )
    identified_estimand = causal_model.identify_effect(
        proceed_when_unidentifiable=True
    )
    estimate = causal_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )

    results: List[RefutationResult] = []

    # ── 1. Placebo treatment ─────────────────────────────────────────────────
    try:
        ref = causal_model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=n_simulations,
        )
        new_ate = float(ref.new_effect)
        p_val   = float(ref.refutation_result.get("p_value", np.nan))
        passed  = p_val > 0.05 if not np.isnan(p_val) else abs(new_ate) < abs(original_ate) * 0.3

        results.append(RefutationResult(
            test_name="Placebo Treatment",
            original_ate=round(original_ate, 5),
            new_ate=round(new_ate, 5),
            p_value=round(p_val, 4) if not np.isnan(p_val) else None,
            passed=passed,
            interpretation=(
                "✅ Robust: ATE collapses near zero when treatment is randomised, "
                "suggesting the original estimate is not spurious."
                if passed else
                "⚠️  Caution: placebo ATE is large relative to original estimate."
            ),
        ))
    except Exception as exc:
        results.append(_error_result("Placebo Treatment", original_ate, str(exc)))

    # ── 2. Random common cause ───────────────────────────────────────────────
    try:
        ref = causal_model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=n_simulations,
        )
        new_ate  = float(ref.new_effect)
        p_val    = float(ref.refutation_result.get("p_value", np.nan))
        rel_diff = abs(new_ate - original_ate) / (abs(original_ate) + 1e-9)
        passed   = rel_diff < 0.20   # < 20 % shift → stable

        results.append(RefutationResult(
            test_name="Random Common Cause",
            original_ate=round(original_ate, 5),
            new_ate=round(new_ate, 5),
            p_value=round(p_val, 4) if not np.isnan(p_val) else None,
            passed=passed,
            interpretation=(
                "✅ Robust: adding a random confounder does not materially "
                "shift the ATE (< 20 % change)."
                if passed else
                f"⚠️  Sensitive: ATE shifts by {rel_diff*100:.1f} % under a "
                "random confounder — consider unmeasured confounding."
            ),
        ))
    except Exception as exc:
        results.append(_error_result("Random Common Cause", original_ate, str(exc)))

    # ── 3. Data subset ───────────────────────────────────────────────────────
    try:
        ref = causal_model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="bootstrap_refuter",
            num_simulations=n_simulations,
            fraction_rows=0.8,
        )
        new_ate  = float(ref.new_effect)
        p_val    = float(ref.refutation_result.get("p_value", np.nan))
        rel_diff = abs(new_ate - original_ate) / (abs(original_ate) + 1e-9)
        passed   = rel_diff < 0.15

        results.append(RefutationResult(
            test_name="Data Subset (Bootstrap)",
            original_ate=round(original_ate, 5),
            new_ate=round(new_ate, 5),
            p_value=round(p_val, 4) if not np.isnan(p_val) else None,
            passed=passed,
            interpretation=(
                "✅ Robust: ATE is stable across 80 % data subsets."
                if passed else
                f"⚠️  Unstable: ATE varies by {rel_diff*100:.1f} % across "
                "subsets — estimate may be sensitive to specific observations."
            ),
        ))
    except Exception as exc:
        results.append(_error_result("Data Subset (Bootstrap)", original_ate, str(exc)))

    return results


def refutations_to_df(results: List[RefutationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Test": r.test_name,
            "Original ATE": r.original_ate,
            "New ATE": r.new_ate,
            "p-value": r.p_value if r.p_value is not None else "—",
            "Result": "✅ PASS" if r.passed else "⚠️  FAIL",
        })
    return pd.DataFrame(rows)


def _error_result(name: str, original_ate: float, err: str) -> RefutationResult:
    return RefutationResult(
        test_name=name,
        original_ate=original_ate,
        new_ate=float("nan"),
        p_value=None,
        passed=False,
        interpretation=f"Error during refutation: {err}",
    )
