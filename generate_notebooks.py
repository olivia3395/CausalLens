"""
Script to generate notebooks/01_lalonde_analysis.ipynb
Run this once: python generate_notebooks.py
"""

import json

lalonde_nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {
   "cell_type": "markdown", "metadata": {}, "id": "title",
   "source": [
    "# 🔍 CausalLens — Lalonde Job Training (NSW) Analysis\n",
    "\n",
    "**Causal Question**: Does participating in a subsidised job-training programme\n",
    "causally increase a worker's 1978 earnings?\n",
    "\n",
    "**Dataset**: Lalonde (1986) NSW experiment — the gold-standard benchmark in causal inference.\n",
    "The NSW was a genuine randomised controlled trial, so we know the *true* ATE.\n",
    "We use it here to verify that our observational methods recover the experimental ground truth.\n",
    "\n",
    "| Variable | Role | Description |\n",
    "|----------|------|-------------|\n",
    "| `treat`  | Treatment | 1 = job training, 0 = control |\n",
    "| `re78`   | Outcome  | 1978 earnings (USD) |\n",
    "| `age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `re74`, `re75` | Confounders | Demographics & prior earnings |"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "setup", "outputs": [],
   "source": [
    "import sys, os, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "sys.path.insert(0, os.path.join(os.getcwd(), '..'))\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "plt.rcParams['font.family'] = 'DejaVu Sans'\n",
    "sns.set_style('whitegrid')\n",
    "print('Libraries loaded ✓')"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s1",
   "source": ["## 1. Load & Explore Data"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "load", "outputs": [],
   "source": [
    "from data.load_data import load_lalonde, dataset_summary, LALONDE_TREATMENT, LALONDE_OUTCOME, LALONDE_CONFOUNDERS\n",
    "\n",
    "df = load_lalonde()\n",
    "print(f'Shape: {df.shape}')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "summary", "outputs": [],
   "source": [
    "summary = dataset_summary(df, LALONDE_TREATMENT, LALONDE_OUTCOME)\n",
    "print('Dataset summary:')\n",
    "for k, v in summary.items():\n",
    "    print(f'  {k:30s}: {v}')"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "eda1", "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
    "\n",
    "# Outcome distribution by treatment\n",
    "for grp, label, color in [(1, 'Treated', '#2ecc71'), (0, 'Control', '#e74c3c')]:\n",
    "    axes[0].hist(df.loc[df['treat']==grp, 're78'], bins=30,\n",
    "                 alpha=0.6, label=label, color=color, edgecolor='white')\n",
    "axes[0].set_title('1978 Earnings by Treatment Status')\n",
    "axes[0].set_xlabel('re78 (USD)')\n",
    "axes[0].legend()\n",
    "\n",
    "# Prior earnings comparison\n",
    "df_melt = df[['treat','re74','re75']].melt(id_vars='treat')\n",
    "sns.boxplot(data=df_melt, x='variable', y='value', hue='treat', ax=axes[1],\n",
    "            palette={0: '#e74c3c', 1: '#2ecc71'})\n",
    "axes[1].set_title('Prior Earnings (re74 / re75)')\n",
    "axes[1].set_xlabel('')\n",
    "\n",
    "# Balance check: education\n",
    "df.groupby('treat')['educ'].plot(kind='kde', ax=axes[2])\n",
    "axes[2].set_title('Education Distribution')\n",
    "axes[2].set_xlabel('Years of Education')\n",
    "axes[2].legend(['Control', 'Treated'])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s2",
   "source": ["## 2. Causal DAG"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "dag", "outputs": [],
   "source": [
    "from causal.discovery import build_causal_graph, plot_causal_graph\n",
    "\n",
    "G = build_causal_graph('lalonde', LALONDE_TREATMENT, LALONDE_OUTCOME)\n",
    "fig = plot_causal_graph(G, LALONDE_TREATMENT, LALONDE_OUTCOME,\n",
    "                        title='Causal DAG — Lalonde NSW')\n",
    "plt.show()\n",
    "print(f'Nodes: {list(G.nodes())}')\n",
    "print(f'Causal path: treat → re78')"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s3",
   "source": ["## 3. Causal Effect Estimation"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "estimation", "outputs": [],
   "source": [
    "from causal.effect_estimation import estimate_all_effects\n",
    "\n",
    "results = estimate_all_effects(\n",
    "    df,\n",
    "    treatment=LALONDE_TREATMENT,\n",
    "    outcome=LALONDE_OUTCOME,\n",
    "    confounders=LALONDE_CONFOUNDERS,\n",
    "    dataset_name='Lalonde NSW',\n",
    "    graph=G,\n",
    ")\n",
    "\n",
    "print(f'\\nNaive ATE         : {results.naive_ate:+.2f}')\n",
    "print(f'PSM ATE           : {results.psm_ate:+.2f}')\n",
    "print(f'IPW ATE           : {results.ipw_ate:+.2f}')\n",
    "print(f'Double ML ATE     : {results.dml_ate:+.2f}  95% CI [{results.dml_ci_lower:.2f}, {results.dml_ate_upper:.2f}]')\n",
    "print(f'Causal Forest ATE : {results.cfdml_ate:+.2f}  95% CI [{results.cfdml_ci_lower:.2f}, {results.cfdml_ci_upper:.2f}]')\n",
    "print(f'\\n(Experimental ground truth from Lalonde 1986: ~$886)')"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "plot_estimates", "outputs": [],
   "source": [
    "summary_df = results.summary_df()\n",
    "display(summary_df)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 3.5))\n",
    "ates = summary_df['ATE'].astype(float)\n",
    "colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in ates]\n",
    "bars = ax.barh(summary_df['Method'], ates, color=colors, height=0.5)\n",
    "ax.axvline(0, color='#333', lw=1.2, ls='--')\n",
    "ax.axvline(results.naive_ate, color='#f39c12', lw=1.5, ls=':', label=f'Naive ATE ({results.naive_ate:+.1f})')\n",
    "ax.axvline(886, color='#8e44ad', lw=1.5, ls='-.', label='Experimental truth (~886)')\n",
    "ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)\n",
    "ax.set_xlabel('ATE (USD)')\n",
    "ax.set_title('Causal Effect Estimates vs Ground Truth', fontweight='bold')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s4",
   "source": ["## 4. Refutation Tests"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "refutation", "outputs": [],
   "source": [
    "from causal.refutation import run_all_refutations, refutations_to_df\n",
    "\n",
    "refutations = run_all_refutations(\n",
    "    df, LALONDE_TREATMENT, LALONDE_OUTCOME, LALONDE_CONFOUNDERS,\n",
    "    original_ate=results.dml_ate or 886.0,\n",
    "    n_simulations=100,\n",
    ")\n",
    "\n",
    "display(refutations_to_df(refutations))\n",
    "\n",
    "for r in refutations:\n",
    "    icon = '✅' if r.passed else '⚠️'\n",
    "    print(f'{icon} {r.test_name}: {r.interpretation}')"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s5",
   "source": ["## 5. LLM Business Report (requires API key)"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "report", "outputs": [],
   "source": [
    "import os\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv(dotenv_path='../.env')\n",
    "\n",
    "from llm.report_generator import generate_full_report\n",
    "\n",
    "dataset_desc = (\n",
    "    'Lalonde NSW Job-Training dataset. We estimate whether participating '\n",
    "    'in a job-training programme causally increases 1978 earnings, '\n",
    "    'controlling for demographics and prior earnings.'\n",
    ")\n",
    "\n",
    "report = generate_full_report(results, refutations, dataset_desc)\n",
    "print(report)"
   ]
  }
 ]
}

ibm_nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {
   "cell_type": "markdown", "metadata": {}, "id": "title",
   "source": [
    "# 🔍 CausalLens — IBM HR Attrition Analysis\n",
    "\n",
    "**Causal Question**: Does earning above-median income causally *reduce* the probability\n",
    "of employee attrition, after controlling for job level, satisfaction, and tenure?\n",
    "\n",
    "| Variable | Role | Description |\n",
    "|----------|------|-------------|\n",
    "| `HighIncome` | Treatment | 1 = above-median monthly income |\n",
    "| `Attrition_num` | Outcome | 1 = employee left, 0 = stayed |\n",
    "| Age, JobLevel, Satisfaction, OverTime, … | Confounders | HR features |"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "setup", "outputs": [],
   "source": [
    "import sys, os, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "sys.path.insert(0, os.path.join(os.getcwd(), '..'))\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "plt.rcParams['figure.dpi'] = 120\n",
    "sns.set_style('whitegrid')\n",
    "print('Libraries loaded ✓')"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s1", "source": ["## 1. Load & Explore Data"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "load", "outputs": [],
   "source": [
    "from data.load_data import load_ibm, dataset_summary, IBM_TREATMENT, IBM_OUTCOME, IBM_CONFOUNDERS\n",
    "\n",
    "df = load_ibm()\n",
    "print(f'Shape: {df.shape}')\n",
    "print(f'Attrition rate: {df[IBM_OUTCOME].mean():.1%}')\n",
    "print(f'High-income rate: {df[IBM_TREATMENT].mean():.1%}')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "eda", "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "\n",
    "# Attrition rate by income group\n",
    "attr_by_inc = df.groupby(IBM_TREATMENT)[IBM_OUTCOME].mean().reset_index()\n",
    "attr_by_inc[IBM_TREATMENT] = attr_by_inc[IBM_TREATMENT].map({0: 'Low Income', 1: 'High Income'})\n",
    "axes[0].bar(attr_by_inc[IBM_TREATMENT], attr_by_inc[IBM_OUTCOME],\n",
    "            color=['#e74c3c','#2ecc71'], edgecolor='white', width=0.5)\n",
    "axes[0].set_ylabel('Attrition Rate')\n",
    "axes[0].set_title('Attrition Rate by Income Group')\n",
    "\n",
    "# Job satisfaction distribution\n",
    "for grp, lbl, c in [(0,'Low Income','#e74c3c'),(1,'High Income','#2ecc71')]:\n",
    "    axes[1].hist(df.loc[df[IBM_TREATMENT]==grp,'JobSatisfaction'],\n",
    "                 bins=4, alpha=0.6, label=lbl, color=c, edgecolor='white')\n",
    "axes[1].set_title('Job Satisfaction by Income Group')\n",
    "axes[1].legend()\n",
    "\n",
    "# Years at company\n",
    "df.boxplot(column='YearsAtCompany', by=IBM_TREATMENT, ax=axes[2])\n",
    "axes[2].set_title('Years at Company by Income Group')\n",
    "axes[2].set_xlabel('HighIncome')\n",
    "plt.suptitle('')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s2", "source": ["## 2. Causal DAG"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "dag", "outputs": [],
   "source": [
    "from causal.discovery import build_causal_graph, plot_causal_graph\n",
    "\n",
    "G = build_causal_graph('ibm', IBM_TREATMENT, IBM_OUTCOME)\n",
    "fig = plot_causal_graph(G, IBM_TREATMENT, IBM_OUTCOME,\n",
    "                        title='Causal DAG — IBM HR Attrition')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s3", "source": ["## 3. Causal Effect Estimation"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "estimation", "outputs": [],
   "source": [
    "from causal.effect_estimation import estimate_all_effects\n",
    "\n",
    "results = estimate_all_effects(\n",
    "    df,\n",
    "    treatment=IBM_TREATMENT,\n",
    "    outcome=IBM_OUTCOME,\n",
    "    confounders=IBM_CONFOUNDERS,\n",
    "    dataset_name='IBM HR Attrition',\n",
    "    graph=G,\n",
    ")\n",
    "\n",
    "print(f'\\nNaive ATE (mean diff)  : {results.naive_ate:+.4f}')\n",
    "print(f'PSM ATE                : {results.psm_ate:+.4f}')\n",
    "print(f'IPW ATE                : {results.ipw_ate:+.4f}')\n",
    "print(f'Double ML ATE          : {results.dml_ate:+.4f}  95% CI [{results.dml_ci_lower:.4f}, {results.dml_ate_upper:.4f}]')\n",
    "print(f'Causal Forest ATE      : {results.cfdml_ate:+.4f}  95% CI [{results.cfdml_ci_lower:.4f}, {results.cfdml_ci_upper:.4f}]')\n",
    "print('\\nInterpretation: a negative ATE means high income REDUCES attrition probability.')"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "plot", "outputs": [],
   "source": [
    "summary_df = results.summary_df()\n",
    "display(summary_df)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 3.5))\n",
    "ates = summary_df['ATE'].astype(float)\n",
    "colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in ates]\n",
    "bars = ax.barh(summary_df['Method'], ates, color=colors, height=0.5)\n",
    "ax.axvline(0, color='#333', lw=1.2, ls='--')\n",
    "ax.axvline(results.naive_ate, color='#f39c12', lw=1.5, ls=':',\n",
    "           label=f'Naive ATE ({results.naive_ate:+.4f})')\n",
    "ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)\n",
    "ax.set_xlabel('ATE (change in P(Attrition))')\n",
    "ax.set_title('Causal vs Naive ATE — IBM HR Attrition', fontweight='bold')\n",
    "ax.legend(fontsize=9)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s4", "source": ["## 4. Refutation Tests"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "refutation", "outputs": [],
   "source": [
    "from causal.refutation import run_all_refutations, refutations_to_df\n",
    "\n",
    "refutations = run_all_refutations(\n",
    "    df, IBM_TREATMENT, IBM_OUTCOME, IBM_CONFOUNDERS,\n",
    "    original_ate=results.dml_ate or results.naive_ate,\n",
    "    n_simulations=100,\n",
    ")\n",
    "\n",
    "display(refutations_to_df(refutations))\n",
    "for r in refutations:\n",
    "    icon = '✅' if r.passed else '⚠️'\n",
    "    print(f'{icon} {r.test_name}: {r.interpretation}')"
   ]
  },
  {
   "cell_type": "markdown", "metadata": {}, "id": "s5", "source": ["## 5. LLM Business Report"]
  },
  {
   "cell_type": "code", "execution_count": None, "metadata": {}, "id": "report", "outputs": [],
   "source": [
    "import os\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv(dotenv_path='../.env')\n",
    "\n",
    "from llm.report_generator import generate_full_report\n",
    "\n",
    "dataset_desc = (\n",
    "    'IBM HR Attrition dataset (n=1470). '\n",
    "    'We investigate whether earning above-median income causally reduces '\n",
    "    'employee attrition probability, controlling for job level, satisfaction, '\n",
    "    'overtime, and tenure.'\n",
    ")\n",
    "\n",
    "report = generate_full_report(results, refutations, dataset_desc)\n",
    "print(report)"
   ]
  }
 ]
}

import json, pathlib
nb_dir = pathlib.Path(__file__).parent / 'notebooks'
nb_dir.mkdir(exist_ok=True)

with open(nb_dir / '01_lalonde_analysis.ipynb', 'w') as f:
    json.dump(lalonde_nb, f, indent=1)

with open(nb_dir / '02_ibm_attrition_analysis.ipynb', 'w') as f:
    json.dump(ibm_nb, f, indent=1)

print('Notebooks generated ✓')
