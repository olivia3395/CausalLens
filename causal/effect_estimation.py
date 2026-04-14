"""
causal/effect_estimation.py
----------------------------
Average Treatment Effect (ATE) estimation via three methods:

  1. Propensity Score Matching     (DoWhy backdoor / PSM)
  2. Inverse Probability Weighting (DoWhy backdoor / IPW)
  3. Double / Debiased ML          (EconML LinearDML with GBM nuisance models)

All three methods are run and results are returned in a CausalResults dataclass
so downstream code (Streamlit app, notebooks, LLM reporter) can consume them
uniformly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# DoWhy
import dowhy
from dowhy import CausalModel

# EconML
from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class CausalResults:
    dataset: str
    treatment: str
    outcome: str
    n_total: int
    n_treated: int
    n_control: int
    naive_ate: float                           # simple mean difference

    psm_ate: Optional[float] = None
    psm_pvalue: Optional[float] = None

    ipw_ate: Optional[float] = None

    dml_ate: Optional[float] = None
    dml_ci_lower: Optional[float] = None
    dml_ate_upper: Optional[float] = None

    cfdml_ate: Optional[float] = None
    cfdml_ci_lower: Optional[float] = None
    cfdml_ci_upper: Optional[float] = None

    errors: List[str] = field(default_factory=list)

    # ---------- helpers -------------------------------------------------------

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "errors"}

    def summary_df(self) -> pd.DataFrame:
        rows = []
        if self.psm_ate is not None:
            rows.append({
                "Method": "Propensity Score Matching (PSM)",
                "ATE": round(self.psm_ate, 5),
                "CI Lower": "—",
                "CI Upper": "—",
                "p-value": f"{self.psm_pvalue:.4f}" if self.psm_pvalue else "—",
            })
        if self.ipw_ate is not None:
            rows.append({
                "Method": "Inverse Probability Weighting (IPW)",
                "ATE": round(self.ipw_ate, 5),
                "CI Lower": "—",
                "CI Upper": "—",
                "p-value": "—",
            })
        if self.dml_ate is not None:
            rows.append({
                "Method": "Double ML (LinearDML + GBM)",
                "ATE": round(self.dml_ate, 5),
                "CI Lower": round(self.dml_ci_lower, 5) if self.dml_ci_lower else "—",
                "CI Upper": round(self.dml_ate_upper, 5) if self.dml_ate_upper else "—",
                "p-value": "—",
            })
        if self.cfdml_ate is not None:
            rows.append({
                "Method": "Causal Forest DML",
                "ATE": round(self.cfdml_ate, 5),
                "CI Lower": round(self.cfdml_ci_lower, 5) if self.cfdml_ci_lower else "—",
                "CI Upper": round(self.cfdml_ci_upper, 5) if self.cfdml_ci_upper else "—",
                "p-value": "—",
            })
        return pd.DataFrame(rows)


# ── Main estimation function ──────────────────────────────────────────────────

def estimate_all_effects(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: List[str],
    dataset_name: str = "dataset",
    graph: Optional[object] = None,   # networkx DiGraph or None
) -> CausalResults:
    """
    Run PSM, IPW, Double ML, and Causal Forest on *df*.

    Parameters
    ----------
    df           : cleaned pandas DataFrame
    treatment    : binary treatment column name
    outcome      : continuous or binary outcome column name
    confounders  : list of confounder column names
    dataset_name : label for the CausalResults object
    graph        : optional nx.DiGraph; if None DoWhy infers from confounders

    Returns
    -------
    CausalResults dataclass with all estimates populated (or error logged).
    """
    # drop rows with NaN in relevant columns
    cols = [treatment, outcome] + confounders
    df = df[cols].dropna().copy()

    n_treated = int(df[treatment].sum())
    n_control = int((df[treatment] == 0).sum())
    naive_ate = float(
        df.loc[df[treatment] == 1, outcome].mean()
        - df.loc[df[treatment] == 0, outcome].mean()
    )

    results = CausalResults(
        dataset=dataset_name,
        treatment=treatment,
        outcome=outcome,
        n_total=len(df),
        n_treated=n_treated,
        n_control=n_control,
        naive_ate=round(naive_ate, 5),
    )

    # ── 1. Build DoWhy model ─────────────────────────────────────────────────
    try:
        if graph is not None:
            import networkx as nx
            # convert nx.DiGraph to GML string for DoWhy
            gml_str = _nx_to_dowhy_gml(graph)
            causal_model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                graph=gml_str,
            )
        else:
            causal_model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders,
            )

        identified_estimand = causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )
    except Exception as exc:
        results.errors.append(f"DoWhy model build failed: {exc}")
        identified_estimand = None
        causal_model = None

    # ── 2. PSM ───────────────────────────────────────────────────────────────
    if identified_estimand is not None:
        try:
            psm_estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_matching",
                target_units="ate",
                method_params={"number_of_matching_attempts": 5},
            )
            results.psm_ate = round(float(psm_estimate.value), 5)
        except Exception as exc:
            results.errors.append(f"PSM failed: {exc}")

    # ── 3. IPW ───────────────────────────────────────────────────────────────
    if identified_estimand is not None:
        try:
            ipw_estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.propensity_score_weighting",
                target_units="ate",
                method_params={"weighting_scheme": "ips_weight"},
            )
            results.ipw_ate = round(float(ipw_estimate.value), 5)
        except Exception as exc:
            results.errors.append(f"IPW failed: {exc}")

    # ── 4. Double ML (LinearDML) ─────────────────────────────────────────────
    try:
        X = df[confounders].values
        T = df[treatment].values.ravel()
        Y = df[outcome].values.ravel()

        model_y = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm",    GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ])
        model_t = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm",    GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
            )),
        ])

        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=5,
            random_state=42,
        )
        dml.fit(Y, T, X=X)

        ate      = float(dml.ate(X))
        ci       = dml.ate_interval(X, alpha=0.05)
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])

        results.dml_ate       = round(ate, 5)
        results.dml_ci_lower  = round(ci_lower, 5)
        results.dml_ate_upper = round(ci_upper, 5)

    except Exception as exc:
        results.errors.append(f"Double ML failed: {exc}")

    # ── 5. Causal Forest DML ─────────────────────────────────────────────────
    try:
        cf = CausalForestDML(
            n_estimators=200,
            min_samples_leaf=10,
            discrete_treatment=True,
            random_state=42,
            cv=5,
        )
        cf.fit(Y, T, X=X)

        cf_ate   = float(cf.ate(X))
        cf_ci    = cf.ate_interval(X, alpha=0.05)
        cf_lower = float(cf_ci[0])
        cf_upper = float(cf_ci[1])

        results.cfdml_ate      = round(cf_ate, 5)
        results.cfdml_ci_lower = round(cf_lower, 5)
        results.cfdml_ci_upper = round(cf_upper, 5)

    except Exception as exc:
        results.errors.append(f"Causal Forest DML failed: {exc}")

    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _nx_to_dowhy_gml(G) -> str:
    """Convert a networkx DiGraph to a DoWhy-compatible GML string."""
    import networkx as nx
    lines = ["graph [", "  directed 1"]
    node_idx = {n: i for i, n in enumerate(G.nodes())}
    for node, idx in node_idx.items():
        lines.append(f'  node [ id {idx} label "{node}" ]')
    for u, v in G.edges():
        lines.append(f'  edge [ source {node_idx[u]} target {node_idx[v]} ]')
    lines.append("]")
    return "\n".join(lines)
