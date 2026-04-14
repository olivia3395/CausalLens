<div align="center">

# 🔍 CausalLens

### LLM-Augmented Causal Inference Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![DoWhy](https://img.shields.io/badge/DoWhy-0.11-1D9E75?style=flat-square)
![EconML](https://img.shields.io/badge/EconML-0.15-185FA5?style=flat-square)
![Claude API](https://img.shields.io/badge/Claude_API-Sonnet_4-7F77DD?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-888780?style=flat-square)

**Causal question answering over tabular data.**
Combines DoWhy, EconML Double ML, and the Claude API to produce
rigorous ATE estimates *and* business-readable reports — end to end.

</div>

---

## Pipeline

```
Raw CSV  →  Causal DAG  →  ATE Estimation  →  Refutation Tests  →  LLM Report
              (DoWhy)     PSM · IPW · DML      Placebo · Boot      Claude API
                           Causal Forest
```

---

## ✨ What It Does

| Step | Method | Library |
|------|--------|---------|
| 1. Causal DAG construction | Domain-knowledge graph | `networkx` · `DoWhy` |
| 2. ATE estimation × 4 methods | PSM · IPW · Double ML · Causal Forest | `dowhy` · `econml` |
| 3. Robustness checks | Placebo · Random confounder · Bootstrap | `dowhy` |
| 4. LLM graph interpretation | DAG → plain English | `anthropic` Claude API |
| 5. Business report generation | Full markdown report | `anthropic` Claude API |
| 6. Interactive UI | Dataset explorer + visualisations | `streamlit` |

---

## 📦 Datasets

### 1. Lalonde NSW Job Training
- **Source**: Lalonde (1986) — classic RCT benchmark in causal inference
- **Treatment**: `treat` — participated in subsidised job training
- **Outcome**: `re78` — 1978 earnings (USD)
- **Ground truth ATE**: ~$886 (from the original experiment)
- **Use case**: Validate that observational methods recover the experimental truth

### 2. IBM HR Employee Attrition
- **Source**: IBM Watson Analytics / AIF360 (n = 1,470)
- **Treatment**: `HighIncome` — monthly income above median
- **Outcome**: `Attrition_num` — employee left the company (1/0)
- **Use case**: Estimate causal impact of compensation on retention

---

## 🚀 Quickstart

```bash
# 1. Clone / unzip
cd CausalLens

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Anthropic API key
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 4a. Run the Streamlit app
streamlit run app/streamlit_app.py

# 4b. Or run the notebooks
jupyter notebook notebooks/
```

---

## 📁 Project Structure

```
CausalLens/
├── data/
│   ├── __init__.py
│   ├── load_data.py          # auto-fetch IBM + Lalonde datasets
│   └── cache/                # local CSV cache (auto-created)
├── causal/
│   ├── __init__.py
│   ├── discovery.py          # build & visualise causal DAGs
│   ├── effect_estimation.py  # PSM · IPW · Double ML · Causal Forest
│   └── refutation.py         # placebo · random confounder · bootstrap
├── llm/
│   ├── __init__.py
│   ├── graph_interpreter.py  # LLM explains the causal DAG
│   └── report_generator.py   # LLM writes a full business report
├── app/
│   └── streamlit_app.py      # interactive Streamlit UI
├── notebooks/
│   ├── 01_lalonde_analysis.ipynb
│   └── 02_ibm_attrition_analysis.ipynb
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧠 Methods Detail

### Propensity Score Matching (PSM)
Matches each treated unit to the most similar control unit based on the
estimated propensity score P(T=1|X).  Implemented via DoWhy's backdoor estimator.

### Inverse Probability Weighting (IPW)
Re-weights the sample so that treated and control groups have the same
covariate distribution.  Uses stabilised IPS weights.

### Double / Debiased Machine Learning (LinearDML)
Removes the effect of confounders via two nuisance models:
- `model_y`: GBM predicts outcome from confounders → residual Ỹ
- `model_t`: GBM predicts treatment from confounders → residual T̃

Then regresses Ỹ on T̃ to obtain the ATE.  Provides valid confidence
intervals via cross-fitting (5-fold CV).

### Causal Forest DML
Non-parametric extension of Double ML using a causal forest as the final
stage estimator.  Estimates heterogeneous treatment effects (CATE) and
reports the average.

### Refutation Tests
| Test | What it checks |
|------|----------------|
| Placebo Treatment | Replace treatment with noise → ATE should collapse to ≈ 0 |
| Random Common Cause | Add a spurious confounder → ATE should be stable |
| Bootstrap (Data Subset) | Re-estimate on 80 % subsets → ATE should be stable |

---

## 📝 Sample LLM Report Output

```
## Executive Summary
Participation in the NSW job-training programme causally increased 1978
earnings by approximately $921 (Double ML, 95% CI [$412, $1,430]).  This
estimate is consistent across PSM ($876), IPW ($904), and Causal Forest
($935), and passes all three refutation tests, giving high confidence
that the effect is not a statistical artefact.

## Business Implications
Policy-makers should continue funding similar job-training programmes...
```

---

## 📊 Resume Bullet

```latex
\item LLM-augmented causal inference pipeline on IBM HR Attrition and
Lalonde datasets; combined DoWhy (PSM, IPW) with EconML Double ML and
Causal Forest (GBM nuisance models) for ATE estimation and refutation
testing (placebo, random confounder, bootstrap); Claude API
auto-generates business reports from causal estimates via a
planner--estimator--reporter workflow; deployed as an interactive
Streamlit app.
```

---

## 🔑 API Key

The LLM features (graph interpretation + report generation) require an
Anthropic API key.  All causal estimation runs without one.

Get a free key at: https://console.anthropic.com

---

## 📚 References

- Lalonde, R. J. (1986). *Evaluating the econometric evaluations of training programs.*
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.*
- Chernozhukov et al. (2018). *Double/debiased machine learning.*
- DoWhy: https://github.com/py-why/dowhy
- EconML: https://github.com/py-why/EconML
