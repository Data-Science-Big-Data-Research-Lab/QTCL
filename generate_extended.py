#!/usr/bin/env python3
"""
Extended summary generators for QTCL experiments.

Produces additional tables, figures, and metrics beyond generate_summary.py:
  - Per-task breakdown tables
  - Correlation and effect-size matrices
  - Pareto fronts and efficiency plots
  - Radar charts, critical difference diagrams
  - DER++ sensitivity heatmaps
  - Quantum-specific analyses

Usage:
    python generate_extended.py --results-dir ./results --paper-dir ../paper
"""

import os
import sys
import glob
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_theme(style="whitegrid", context="paper")
except ImportError:
    HAS_SEABORN = False

# Import existing helpers if available
try:
    import statistical_tests as st
    HAS_STATSTESTS = True
except ImportError:
    HAS_STATSTESTS = False


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(results_dir, "**", "runs.csv"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No runs.csv found under {results_dir}")
    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["run_id"])
    if "study" not in df.columns:
        df["study"] = "main"
    df["study"] = df["study"].fillna("main")
    return df


def safe_mean_std(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    m = np.mean(values)
    s = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return m, s


def fmt_pm(mean, std, pct: bool = True, dec: int = 2) -> str:
    if pd.isna(mean):
        return "---"
    if pct:
        return f"${mean*100:.{dec}f} \\pm {std*100:.{dec}f}$"
    return f"${mean:.{dec}f} \\pm {std:.{dec}f}$"


def write_tex(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[TABLE] {path}")


def save_fig(fig, path: str, dpi: int = 200) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG]   {path}")


# ── Derived metrics ───────────────────────────────────────────────────────────

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns for downstream analysis."""
    if "train_time_s" in df.columns and "n_tasks" in df.columns:
        df["time_per_task_s"] = df["train_time_s"] / df["n_tasks"].clip(lower=1)
    if "AA" in df.columns and "n_params_head" in df.columns:
        df["acc_per_param"] = df["AA"] / df["n_params_head"].clip(lower=1)
    if "AA" in df.columns and "train_time_s" in df.columns:
        df["acc_per_sec"] = df["AA"] / df["train_time_s"].clip(lower=1)
    if "AA" in df.columns and "energy_kwh" in df.columns:
        df["acc_per_kwh"] = df["AA"] / df["energy_kwh"].clip(lower=1e-9)
    if "co2_kg" in df.columns and "AA" in df.columns:
        df["co2_per_acc"] = df["co2_kg"] / df["AA"].clip(lower=1e-9)
    if "AF" in df.columns and "AA" in df.columns and "BWT" in df.columns:
        df["cl_score"] = (df["AA"] - df["AF"] + np.maximum(0, df["BWT"])) / 2.0
    return df


# ── Tables ────────────────────────────────────────────────────────────────────

def table_per_task_accuracy(df: pd.DataFrame, tables_dir: str) -> None:
    """LaTeX table of per-task final accuracy (mean ± std over seeds)."""
    task_cols = [c for c in df.columns if c.startswith("task_") and c.endswith("_final_acc")]
    if not task_cols:
        print("[TABLE] No per-task accuracy columns found.")
        return
    rows = []
    for (ds, bb, hd), grp in df.groupby(["dataset", "backbone", "head"]):
        cells = []
        for c in task_cols:
            vals = grp[c].dropna()
            if len(vals) > 0:
                m, s = safe_mean_std(vals)
                cells.append(fmt_pm(m, s))
            else:
                cells.append("---")
        rows.append({
            "Dataset": ds.replace("_", " "),
            "Backbone": bb.replace("_", "\\_"),
            "Head": hd.replace("_", "\\_"),
            **{f"Task {i+1}": cells[i] for i in range(len(cells))},
        })
    if not rows:
        return
    result = pd.DataFrame(rows)
    cols = ["Dataset", "Backbone", "Head"] + [f"Task {i+1}" for i in range(len(task_cols))]
    tex = ["\\begin{tabular}{lll" + "c" * len(task_cols) + "}", "\\toprule"]
    tex.append(" & ".join(cols) + " \\\\")
    tex.append("\\midrule")
    for _, row in result.iterrows():
        tex.append(" & ".join(str(row[c]) for c in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "per_task_accuracy.tex"), tex)


def table_ranking(df: pd.DataFrame, tables_dir: str) -> None:
    """Rank each configuration by AA, AF, BWT per dataset."""
    if not all(c in df.columns for c in ["AA", "AF", "BWT"]):
        return
    rows = []
    for ds, grp in df.groupby("dataset"):
        means = grp.groupby(["backbone", "head"])[["AA", "AF", "BWT"]].mean().reset_index()
        means["rank_AA"] = means["AA"].rank(ascending=False).astype(int)
        means["rank_AF"] = means["AF"].rank(ascending=True).astype(int)
        means["rank_BWT"] = means["BWT"].rank(ascending=False).astype(int)
        for _, r in means.iterrows():
            rows.append({
                "Dataset": ds.replace("_", " "),
                "Backbone": r["backbone"].replace("_", "\\_"),
                "Head": r["head"].replace("_", "\\_"),
                "AA rank": r["rank_AA"],
                "AF rank": r["rank_AF"],
                "BWT rank": r["rank_BWT"],
            })
    if not rows:
        return
    result = pd.DataFrame(rows)
    tex = ["\\begin{tabular}{lllccc}", "\\toprule"]
    tex.append("Dataset & Backbone & Head & AA rank & AF rank & BWT rank \\\\")
    tex.append("\\midrule")
    for _, row in result.iterrows():
        tex.append(f"{row['Dataset']} & {row['Backbone']} & {row['Head']} & {row['AA rank']} & {row['AF rank']} & {row['BWT rank']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "ranking.tex"), tex)


def table_correlation(df: pd.DataFrame, tables_dir: str) -> None:
    """LaTeX table of Pearson correlations between metrics."""
    numeric_cols = [c for c in ["AA", "AF", "BWT", "FWT", "train_time_s",
                                 "energy_kwh", "co2_kg", "n_params_head"] if c in df.columns]
    if len(numeric_cols) < 2:
        return
    corr = df[numeric_cols].corr(method="pearson")
    tex = ["\\begin{tabular}{l" + "c" * len(numeric_cols) + "}", "\\toprule"]
    tex.append(" & " + " & ".join(numeric_cols) + " \\\\")
    tex.append("\\midrule")
    for col in numeric_cols:
        vals = [f"{corr.loc[col, c]:.2f}" if not pd.isna(corr.loc[col, c]) else "---" for c in numeric_cols]
        tex.append(col + " & " + " & ".join(vals) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "correlation_matrix.tex"), tex)


def table_effect_size(df: pd.DataFrame, tables_dir: str) -> None:
    """Cohen's d matrix for all pairwise head comparisons."""
    if not HAS_STATSTESTS or "AA" not in df.columns:
        return
    heads = sorted(df["head"].unique())
    if len(heads) < 2:
        return
    tex = ["\\begin{tabular}{l" + "c" * len(heads) + "}", "\\toprule"]
    tex.append(" & " + " & ".join(h.replace('_', '\\_') for h in heads) + " \\\\")
    tex.append("\\midrule")
    for h1 in heads:
        cells = []
        for h2 in heads:
            if h1 == h2:
                cells.append("---")
                continue
            a = df[df["head"] == h1]["AA"].dropna().values
            b = df[df["head"] == h2]["AA"].dropna().values
            if len(a) > 1 and len(b) > 1:
                d = st.cohens_d(a, b)
                cells.append(f"{d:.2f}")
            else:
                cells.append("---")
        tex.append(h1.replace('_', '\\_') + " & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "effect_size_matrix.tex"), tex)


def table_best_config_per_study(df: pd.DataFrame, tables_dir: str) -> None:
    """Best configuration (highest AA) per study."""
    rows = []
    for study, grp in df.groupby("study"):
        if grp["AA"].isna().all():
            continue
        best = grp.loc[grp["AA"].idxmax()]
        rows.append({
            "Study": study.replace("_", "\\_"),
            "Run ID": best.get("run_id", "?").replace("_", "\\_"),
            "Dataset": best.get("dataset", "?").replace("_", " "),
            "Backbone": best.get("backbone", "?").replace("_", "\\_"),
            "Head": best.get("head", "?").replace("_", "\\_"),
            "AA": f"{best['AA']*100:.2f}\\%" if not pd.isna(best.get("AA")) else "?",
            "AF": f"{best['AF']*100:.2f}\\%" if not pd.isna(best.get("AF")) else "?",
        })
    if not rows:
        return
    tex = ["\\begin{tabular}{llllllc}", "\\toprule"]
    tex.append("Study & Run ID & Dataset & Backbone & Head & AA & AF \\\\")
    tex.append("\\midrule")
    for r in rows:
        tex.append(f"{r['Study']} & {r['Run ID']} & {r['Dataset']} & {r['Backbone']} & {r['Head']} & {r['AA']} & {r['AF']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "best_config_per_study.tex"), tex)


def table_computational_budget(df: pd.DataFrame, tables_dir: str) -> None:
    """Total kWh, CO2, and time per study."""
    rows = []
    for study, grp in df.groupby("study"):
        row = {"Study": study, "Runs": len(grp)}
        for col, label in [("train_time_s", "Time (h)"), ("energy_kwh", "Energy (kWh)"), ("co2_kg", "CO2 (kg)")]:
            if col in grp.columns:
                total = grp[col].sum()
                if col == "train_time_s":
                    total /= 3600.0
                row[label] = f"{total:.2f}"
            else:
                row[label] = "---"
        rows.append(row)
    if not rows:
        return
    tex = ["\\begin{tabular}{lccc}", "\\toprule"]
    tex.append("Study & Time (h) & Energy (kWh) & CO2 (kg) \\\\")
    tex.append("\\midrule")
    for r in rows:
        study_esc = r['Study'].replace('_', '\\_')
        tex.append(f"{study_esc} & {r['Time (h)']} & {r['Energy (kWh)']} & {r['CO2 (kg)']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "computational_budget.tex"), tex)


def _extract_alpha_beta_from_run_id(run_id: str):
    """Parse alpha and beta values from a DER++ sensitivity run_id."""
    # Format: {dataset}_{bb}_{hd}_a{alpha}_b{beta}_{seed}
    import re
    m = re.search(r'_a([0-9.]+)_b([0-9.]+)_', str(run_id))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def table_derpp_sensitivity(df: pd.DataFrame, tables_dir: str) -> None:
    """Heatmap-style LaTeX table for DER++ alpha×beta grid."""
    sub = df[df.get("study", pd.Series()) == "derpp_sensitivity"] if "study" in df.columns else pd.DataFrame()
    if sub.empty:
        return
    # Try columns first, then fallback to run_id parsing
    if "alpha" not in sub.columns or "beta" not in sub.columns:
        parsed = sub["run_id"].apply(_extract_alpha_beta_from_run_id)
        sub = sub.copy()
        sub["alpha"] = parsed.apply(lambda x: x[0] if x[0] is not None else None)
        sub["beta"] = parsed.apply(lambda x: x[1] if x[1] is not None else None)
        if sub["alpha"].isna().all():
            print("[TABLE] DER++ sensitivity: alpha/beta columns not found in runs.csv and could not be parsed from run_id.")
            return
    pivot = sub.groupby(["alpha", "beta"])["AA"].agg(["mean", "std"]).reset_index()
    alphas = sorted(pivot["alpha"].unique())
    betas = sorted(pivot["beta"].unique())
    tex = ["\\begin{tabular}{l" + "c" * len(betas) + "}", "\\toprule"]
    tex.append("alpha / beta & " + " & ".join(str(b) for b in betas) + " \\\\")
    tex.append("\\midrule")
    for a in alphas:
        cells = []
        for b in betas:
            row = pivot[(pivot["alpha"] == a) & (pivot["beta"] == b)]
            if not row.empty:
                m = row["mean"].values[0]
                s = row["std"].values[0]
                cells.append(fmt_pm(m, s))
            else:
                cells.append("---")
        tex.append(f"{a} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex(os.path.join(tables_dir, "derpp_sensitivity.tex"), tex)


def plot_violin_metrics(df: pd.DataFrame, figures_dir: str) -> None:
    """Violin plots for AA, AF, BWT distributions by head."""
    if not HAS_SEABORN or "head" not in df.columns:
        return
    metrics = [("AA", "Average Accuracy"), ("AF", "Average Forgetting"), ("BWT", "Backward Transfer")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        sns.violinplot(data=df, x="head", y=col, ax=ax, inner="box", cut=0)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(title)
        ax.set_ylabel(f"{col} (fraction)")
    save_fig(fig, os.path.join(figures_dir, "violin_metrics.pdf"))


def plot_radar_heads(df: pd.DataFrame, figures_dir: str) -> None:
    """Radar chart showing multi-metric profile per head (averaged over seeds)."""
    if "head" not in df.columns:
        return
    metrics = [c for c in ["AA", "AF", "BWT", "final_f1_mean", "final_roc_auc_mean"] if c in df.columns]
    if len(metrics) < 3:
        return
    grp = df.groupby("head")[metrics].mean().reset_index()
    norm = grp[metrics].copy()
    for c in metrics:
        lo, hi = norm[c].min(), norm[c].max()
        if hi > lo:
            norm[c] = (norm[c] - lo) / (hi - lo)
        else:
            norm[c] = 0.5
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in grp.iterrows():
        hd = row["head"]
        vals = [norm.loc[row.name, c] for c in metrics] + [norm.loc[row.name, metrics[0]]]
        ax.plot(angles, vals, label=hd)
        ax.fill(angles, vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save_fig(fig, os.path.join(figures_dir, "radar_heads.pdf"))


def plot_pareto_energy_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    """Pareto front: energy vs accuracy."""
    if "energy_kwh" not in df.columns or "AA" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for hd, grp in df.groupby("head"):
        ax.scatter(grp["energy_kwh"], grp["AA"], label=hd, alpha=0.7, s=60)
    ax.set_xlabel("Energy (kWh)")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Pareto Front: Energy vs Accuracy")
    ax.legend()
    save_fig(fig, os.path.join(figures_dir, "pareto_energy_accuracy.pdf"))


def plot_pareto_time_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    """Pareto front: time vs accuracy."""
    if "train_time_s" not in df.columns or "AA" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for hd, grp in df.groupby("head"):
        ax.scatter(grp["train_time_s"] / 60.0, grp["AA"], label=hd, alpha=0.7, s=60)
    ax.set_xlabel("Training Time (min)")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Pareto Front: Time vs Accuracy")
    ax.legend()
    save_fig(fig, os.path.join(figures_dir, "pareto_time_accuracy.pdf"))


def plot_accuracy_retention_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    """Heatmap: final accuracy on each task (columns) per method (rows)."""
    task_cols = [c for c in df.columns if c.startswith("task_") and c.endswith("_final_acc")]
    if not task_cols or "head" not in df.columns:
        return
    pivot = df.groupby("head")[[c for c in task_cols]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Accuracy"}, ax=ax)
    ax.set_xlabel("Task")
    ax.set_ylabel("Head")
    ax.set_title("Per-Task Accuracy Retention Heatmap")
    save_fig(fig, os.path.join(figures_dir, "accuracy_retention_heatmap.pdf"))


def plot_parameter_vs_accuracy(df: pd.DataFrame, figures_dir: str) -> None:
    """Scatter: parameter count vs AA."""
    if "n_params_head" not in df.columns or "AA" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for hd, grp in df.groupby("head"):
        ax.scatter(grp["n_params_head"], grp["AA"], label=hd, alpha=0.7, s=60)
    ax.set_xlabel("Trainable Parameters (head)")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Parameters vs Accuracy")
    ax.set_xscale("log")
    ax.legend()
    save_fig(fig, os.path.join(figures_dir, "params_vs_accuracy.pdf"))


def plot_quantum_overhead(df: pd.DataFrame, figures_dir: str) -> None:
    """Bar chart: mean training time ratio quantum/classical per dataset."""
    if "train_time_s" not in df.columns or "head" not in df.columns:
        return
    classical = df[df["head"].isin(["mlp", "linear", "mlp_deep"])].groupby("dataset")["train_time_s"].mean()
    quantum = df[df["head"].str.startswith("qk")].groupby("dataset")["train_time_s"].mean()
    common = classical.index.intersection(quantum.index)
    if len(common) == 0:
        return
    ratios = quantum[common] / classical[common]
    fig, ax = plt.subplots(figsize=(6, 4))
    ratios.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Quantum / Classical Time Ratio")
    ax.set_title("Quantum Overhead Factor")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, os.path.join(figures_dir, "quantum_overhead.pdf"))


def plot_noise_resilience(df: pd.DataFrame, figures_dir: str) -> None:
    """Noise resilience score: AA_noisy / AA_ideal per (dataset, backbone)."""
    if "head" not in df.columns or "AA" not in df.columns:
        return
    ideal = df[df["head"] == "qk_ideal"].groupby(["dataset", "backbone"])["AA"].mean()
    noisy = df[df["head"] == "qk_noisy"].groupby(["dataset", "backbone"])["AA"].mean()
    common = ideal.index.intersection(noisy.index)
    if len(common) == 0:
        return
    ratio = noisy[common] / ideal[common]
    fig, ax = plt.subplots(figsize=(6, 4))
    ratio.plot(kind="bar", ax=ax, color="coral")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("AA_noisy / AA_ideal")
    ax.set_title("Noise Resilience Score")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, os.path.join(figures_dir, "noise_resilience.pdf"))


def plot_derpp_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    """Heatmap of AA as function of DER++ alpha and beta."""
    sub = df[df.get("study", pd.Series()) == "derpp_sensitivity"] if "study" in df.columns else pd.DataFrame()
    if sub.empty:
        return
    if "alpha" not in sub.columns or "beta" not in sub.columns:
        parsed = sub["run_id"].apply(_extract_alpha_beta_from_run_id)
        sub = sub.copy()
        sub["alpha"] = parsed.apply(lambda x: x[0] if x[0] is not None else None)
        sub["beta"] = parsed.apply(lambda x: x[1] if x[1] is not None else None)
        if sub["alpha"].isna().all():
            return
    pivot = sub.pivot_table(index="alpha", columns="beta", values="AA", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "AA"}, ax=ax)
    ax.set_title("DER++ Sensitivity: AA vs Alpha/Beta")
    save_fig(fig, os.path.join(figures_dir, "derpp_sensitivity_heatmap.pdf"))


def plot_ansatz_radar(df: pd.DataFrame, figures_dir: str) -> None:
    """Radar chart comparing all ansatz variants (arch_variants study)."""
    av = df[df.get("study", pd.Series()) == "arch_variants"] if "study" in df.columns else pd.DataFrame()
    if av.empty:
        return
    metrics = [c for c in ["AA", "AF", "BWT", "final_f1_mean", "final_roc_auc_mean"] if c in av.columns]
    if len(metrics) < 3:
        return
    grp = av.groupby("head")[metrics].mean().reset_index()
    norm = grp[metrics].copy()
    for c in metrics:
        lo, hi = norm[c].min(), norm[c].max()
        if hi > lo:
            norm[c] = (norm[c] - lo) / (hi - lo)
        else:
            norm[c] = 0.5
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for _, row in grp.iterrows():
        hd = row["head"]
        vals = [norm.loc[row.name, c] for c in metrics] + [norm.loc[row.name, metrics[0]]]
        ax.plot(angles, vals, label=hd)
        ax.fill(angles, vals, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize="small")
    save_fig(fig, os.path.join(figures_dir, "ansatz_radar.pdf"))


def plot_task_difficulty(df: pd.DataFrame, figures_dir: str) -> None:
    """Bar chart ranking tasks by mean final accuracy across all methods."""
    task_cols = [c for c in df.columns if c.startswith("task_") and c.endswith("_final_acc")]
    if not task_cols:
        return
    means = df[task_cols].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot(kind="barh", ax=ax, color="teal")
    ax.set_xlabel("Mean Final Accuracy")
    ax.set_title("Task Difficulty Ranking")
    save_fig(fig, os.path.join(figures_dir, "task_difficulty.pdf"))


def plot_quantum_advantage(df: pd.DataFrame, figures_dir: str) -> None:
    """Bar chart: (AA_quantum - AA_classical) / AA_classical per dataset."""
    if "head" not in df.columns or "AA" not in df.columns:
        return
    classical = df[df["head"].isin(["mlp", "linear", "mlp_deep"])].groupby("dataset")["AA"].mean()
    quantum = df[df["head"].str.startswith("qk")].groupby("dataset")["AA"].mean()
    common = classical.index.intersection(quantum.index)
    if len(common) == 0:
        return
    gap = (quantum[common] - classical[common]) / classical[common]
    fig, ax = plt.subplots(figsize=(6, 4))
    gap.plot(kind="bar", ax=ax, color="mediumpurple")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("(Quantum - Classical) / Classical")
    ax.set_title("Quantum Advantage Gap")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, os.path.join(figures_dir, "quantum_advantage.pdf"))


def plot_critical_difference(df: pd.DataFrame, figures_dir: str) -> None:
    """Simplified critical difference bar chart (Nemenyi-style)."""
    if not HAS_STATSTESTS or "AA" not in df.columns or "head" not in df.columns:
        return
    heads = sorted(df["head"].unique())
    if len(heads) < 2:
        return
    # Compute mean rank per dataset
    ranks = []
    for ds, grp in df.groupby("dataset"):
        means = grp.groupby("head")["AA"].mean()
        r = means.rank(ascending=False)
        ranks.append(r)
    if not ranks:
        return
    mean_ranks = pd.concat(ranks, axis=1).mean(axis=1).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_ranks.plot(kind="barh", ax=ax, color="slategray")
    ax.set_xlabel("Mean Rank (AA)")
    ax.set_title("Critical Difference Diagram (Mean Rank)")
    save_fig(fig, os.path.join(figures_dir, "critical_difference.pdf"))


def plot_energy_scaling_qubits(df: pd.DataFrame, figures_dir: str) -> None:
    """Log-log plot of energy vs n_qubits (ablation data)."""
    ab = df[df.get("study", pd.Series()) == "ablation"] if "study" in df.columns else pd.DataFrame()
    if ab.empty or "n_qubits" not in ab.columns or "energy_kwh" not in ab.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for hd, grp in ab.groupby("head"):
        sub = grp.groupby("n_qubits")["energy_kwh"].mean().reset_index()
        ax.plot(sub["n_qubits"], sub["energy_kwh"], marker="o", label=hd)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Energy Scaling with Qubits")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    save_fig(fig, os.path.join(figures_dir, "energy_scaling_qubits.pdf"))


def plot_energy_scaling_depth(df: pd.DataFrame, figures_dir: str) -> None:
    """Log-log plot of energy vs depth (ablation data)."""
    ab = df[df.get("study", pd.Series()) == "ablation"] if "study" in df.columns else pd.DataFrame()
    if ab.empty or "depth" not in ab.columns or "energy_kwh" not in ab.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    for hd, grp in ab.groupby("head"):
        sub = grp.groupby("depth")["energy_kwh"].mean().reset_index()
        ax.plot(sub["depth"], sub["energy_kwh"], marker="s", label=hd)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Energy Scaling with Depth")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    save_fig(fig, os.path.join(figures_dir, "energy_scaling_depth.pdf"))


# ── Main orchestrator ─────────────────────────────────────────────────────────

def generate_all_extended(results_dir: str, paper_dir: str) -> None:
    tables_dir = os.path.join(paper_dir, "tables")
    figures_dir = os.path.join(paper_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    try:
        df = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"[EXTENDED] {e}")
        return

    df = add_derived_metrics(df)
    print(f"[EXTENDED] Loaded {len(df)} runs. Generating extended outputs...")

    # Tables
    table_per_task_accuracy(df, tables_dir)
    table_ranking(df, tables_dir)
    table_correlation(df, tables_dir)
    table_effect_size(df, tables_dir)
    table_best_config_per_study(df, tables_dir)
    table_computational_budget(df, tables_dir)
    table_derpp_sensitivity(df, tables_dir)

    # Figures
    plot_violin_metrics(df, figures_dir)
    plot_radar_heads(df, figures_dir)
    plot_pareto_energy_accuracy(df, figures_dir)
    plot_pareto_time_accuracy(df, figures_dir)
    plot_accuracy_retention_heatmap(df, figures_dir)
    plot_parameter_vs_accuracy(df, figures_dir)
    plot_quantum_overhead(df, figures_dir)
    plot_noise_resilience(df, figures_dir)
    plot_derpp_heatmap(df, figures_dir)
    plot_ansatz_radar(df, figures_dir)
    plot_task_difficulty(df, figures_dir)
    plot_quantum_advantage(df, figures_dir)
    plot_critical_difference(df, figures_dir)
    plot_energy_scaling_qubits(df, figures_dir)
    plot_energy_scaling_depth(df, figures_dir)

    print("[EXTENDED] All extended outputs generated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QTCL Extended Summary Generator")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--paper-dir", default="../paper")
    args = p.parse_args()
    generate_all_extended(args.results_dir, args.paper_dir)
