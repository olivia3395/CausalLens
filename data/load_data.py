"""
data/load_data.py
-----------------
Loaders for the two benchmark datasets used in CausalLens:
  1. IBM HR Employee Attrition
  2. Lalonde Job-Training (NSW)

Both are fetched from stable public URLs / pip-installable packages so
the project runs offline after the first download.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "cache"
DATA_DIR.mkdir(exist_ok=True)

IBM_URL = (
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360"
    "/master/data/emp_attrition.csv"
)
IBM_CACHE = DATA_DIR / "ibm_attrition.csv"
LALONDE_CACHE = DATA_DIR / "lalonde.csv"


# ── IBM HR Attrition ─────────────────────────────────────────────────────────

IBM_TREATMENT = "HighIncome"          # engineered binary treatment
IBM_OUTCOME   = "Attrition_num"       # 1 = left, 0 = stayed
IBM_CONFOUNDERS = [
    "Age", "JobLevel", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "WorkLifeBalance", "JobSatisfaction",
    "EnvironmentSatisfaction", "RelationshipSatisfaction",
    "NumCompaniesWorked", "TotalWorkingYears", "TrainingTimesLastYear",
    "OverTime_num", "MaritalStatus_num",
]


def load_ibm(force_download: bool = False) -> pd.DataFrame:
    """
    Load the IBM HR Attrition dataset.

    Returns a cleaned DataFrame with engineered features ready for causal
    analysis.  Key columns:
        - HighIncome       : 1 if MonthlyIncome > median, else 0  (treatment)
        - Attrition_num    : 1 = employee left, 0 = stayed        (outcome)
        - <IBM_CONFOUNDERS>: pre-selected confounder set
    """
    if IBM_CACHE.exists() and not force_download:
        df = pd.read_csv(IBM_CACHE)
    else:
        try:
            df = pd.read_csv(IBM_URL)
            df.to_csv(IBM_CACHE, index=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download IBM dataset from {IBM_URL}.\n"
                "Check your internet connection, or place the CSV manually at "
                f"{IBM_CACHE}."
            ) from exc

    df = _clean_ibm(df)
    return df


def _clean_ibm(df: pd.DataFrame) -> pd.DataFrame:
    # binary outcome
    df["Attrition_num"] = (df["Attrition"] == "Yes").astype(int)

    # binary treatment: above-median monthly income
    median_income = df["MonthlyIncome"].median()
    df["HighIncome"] = (df["MonthlyIncome"] > median_income).astype(int)

    # encode categoricals needed for confounders
    df["OverTime_num"] = (df["OverTime"] == "Yes").astype(int)
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
    df["MaritalStatus_num"] = df["MaritalStatus"].map(marital_map).fillna(0)

    # drop columns with zero variance (e.g., EmployeeCount, StandardHours)
    df = df.loc[:, df.nunique() > 1]

    return df


# ── Lalonde Job-Training ─────────────────────────────────────────────────────

LALONDE_TREATMENT   = "treat"
LALONDE_OUTCOME     = "re78"
LALONDE_CONFOUNDERS = [
    "age", "educ", "black", "hisp", "married", "nodegree", "re74", "re75",
]


def load_lalonde(force_download: bool = False) -> pd.DataFrame:
    """
    Load the Lalonde (NSW) job-training dataset.

    Tries causaldata first; falls back to a stable GitHub mirror.

    Key columns:
        - treat  : 1 = received job training, 0 = control   (treatment)
        - re78   : 1978 earnings in USD                      (outcome)
        - age, educ, black, hisp, married, nodegree, re74, re75 : confounders
    """
    if LALONDE_CACHE.exists() and not force_download:
        return pd.read_csv(LALONDE_CACHE)

    # try causaldata package
    try:
        from causaldata import lalonde  # type: ignore
        df = lalonde.load_data()
        df.to_csv(LALONDE_CACHE, index=False)
        return df
    except Exception:
        pass

    # fallback: GitHub mirror
    fallback_url = (
        "https://raw.githubusercontent.com/matheusfacure/"
        "python-causality-handbook/master/causal-inference-for-the-brave-and-true"
        "/data/lalonde_nsw.csv"
    )
    try:
        df = pd.read_csv(fallback_url)
        # normalise column names
        df.columns = [c.lower() for c in df.columns]
        df.to_csv(LALONDE_CACHE, index=False)
        return df
    except Exception as exc:
        raise RuntimeError(
            "Could not load Lalonde dataset via causaldata or GitHub mirror.\n"
            "Install causaldata:  pip install causaldata"
        ) from exc


# ── helpers ──────────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame, treatment: str, outcome: str) -> dict:
    """Return a quick descriptive summary dict."""
    n_treated = int(df[treatment].sum())
    n_control = int((df[treatment] == 0).sum())
    outcome_mean_t = float(df.loc[df[treatment] == 1, outcome].mean())
    outcome_mean_c = float(df.loc[df[treatment] == 0, outcome].mean())
    naive_diff = outcome_mean_t - outcome_mean_c

    return {
        "n_total": len(df),
        "n_treated": n_treated,
        "n_control": n_control,
        "outcome_mean_treated": round(outcome_mean_t, 4),
        "outcome_mean_control": round(outcome_mean_c, 4),
        "naive_ate": round(naive_diff, 4),
    }
