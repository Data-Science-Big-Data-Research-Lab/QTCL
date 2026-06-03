"""
Statistical tests for comparing methods across seeds.

Implements:
    - Wilcoxon signed-rank test (paired comparison, two methods)
    - Mann-Whitney U test (unpaired comparison, two methods)
    - Friedman test (paired comparison, ≥3 methods)
    - Kruskal-Wallis H test (unpaired, ≥3 methods)
    - Cohen's d effect size
    - Bootstrap 95% confidence intervals
    - Pairwise p-value matrix with Holm-Bonferroni correction
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


# ── Pairwise tests ────────────────────────────────────────────────────────────

def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired non-parametric test. Returns statistic and p-value."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.all(a == b):
        return {"statistic": float("nan"), "p_value": 1.0, "n": len(a)}
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", correction=False, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p), "n": len(a)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "n": len(a)}


def mann_whitney(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Unpaired non-parametric test."""
    try:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p), "n_a": len(a), "n_b": len(b)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "n_a": len(a), "n_b": len(b)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (positive = a > b). Pooled SD."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


# ── Multi-method tests ────────────────────────────────────────────────────────

def friedman(*samples: np.ndarray) -> Dict[str, float]:
    """Paired non-parametric test for ≥3 groups."""
    try:
        stat, p = stats.friedmanchisquare(*samples)
        return {"statistic": float(stat), "p_value": float(p), "k": len(samples), "n": len(samples[0])}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "k": len(samples), "n": len(samples[0])}


def kruskal_wallis(*samples: np.ndarray) -> Dict[str, float]:
    """Unpaired non-parametric test for ≥3 groups."""
    try:
        stat, p = stats.kruskal(*samples)
        return {"statistic": float(stat), "p_value": float(p), "k": len(samples)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "k": len(samples)}


# ── Confidence intervals ──────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap percentile CI. Returns (mean, lower, upper).

    NOTE: This function uses its own independent random seed (default 0)
    for reproducibility of the bootstrap resampling. It does NOT use the
    run seed from the experiment. This is intentional because bootstrap
    is a post-hoc analysis step, not part of the training pipeline.
    """
    values = np.asarray(values)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(values.mean()), float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))


# ── Multiple comparison correction ────────────────────────────────────────────

def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Adjusted p-values via Holm-Bonferroni (control of FWER)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        v = (n - rank) * p[idx]
        running_max = max(running_max, v)
        adj[idx] = min(running_max, 1.0)
    return adj.tolist()


def pairwise_pvalue_matrix(
    df: pd.DataFrame, group_col: str, value_col: str,
    paired: bool = True, correction: str = "holm"
) -> pd.DataFrame:
    """
    Compute pairwise Wilcoxon (paired) or Mann-Whitney (unpaired) p-values
    between every pair of groups.

    Optionally adjust with Holm-Bonferroni.
    """
    groups = sorted(df[group_col].unique())
    n = len(groups)
    raw = np.full((n, n), np.nan)
    pvals_flat = []
    pairs = []

    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            if i == j:
                raw[i, j] = 1.0
                continue
            if i < j:
                a = df[df[group_col] == gi][value_col].values
                b = df[df[group_col] == gj][value_col].values
                if paired:
                    n_min = min(len(a), len(b))
                    res = wilcoxon_signed_rank(a[:n_min], b[:n_min])
                else:
                    res = mann_whitney(a, b)
                raw[i, j] = raw[j, i] = res["p_value"]
                pvals_flat.append(res["p_value"])
                pairs.append((i, j))

    if correction == "holm" and pvals_flat:
        adj = holm_bonferroni(pvals_flat)
        for (i, j), pa in zip(pairs, adj):
            raw[i, j] = raw[j, i] = pa

    return pd.DataFrame(raw, index=groups, columns=groups)


# ── Compact reports ───────────────────────────────────────────────────────────

def method_comparison_report(
    df: pd.DataFrame, method_col: str, metric_col: str,
    seed_col: str = "seed", paired: bool = True
) -> pd.DataFrame:
    """
    Return one row per pair of methods with:
        method_a, method_b, mean_a, mean_b, diff, cohens_d,
        wilcoxon_stat, p_value, p_value_adj, significant_005
    """
    methods = sorted(df[method_col].unique())
    rows = []
    raw_p = []
    keys  = []

    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            a = df[df[method_col] == m1].sort_values(seed_col)[metric_col].values
            b = df[df[method_col] == m2].sort_values(seed_col)[metric_col].values
            n_min = min(len(a), len(b))
            a, b = a[:n_min], b[:n_min]
            if paired:
                res = wilcoxon_signed_rank(a, b)
            else:
                res = mann_whitney(a, b)
            d = cohens_d(a, b)
            rows.append({
                "method_a":     m1,
                "method_b":     m2,
                "mean_a":       float(a.mean()) if len(a) else np.nan,
                "mean_b":       float(b.mean()) if len(b) else np.nan,
                "diff":         float(a.mean() - b.mean()) if len(a) and len(b) else np.nan,
                "cohens_d":     d,
                "test_stat":    res["statistic"],
                "p_value":      res["p_value"],
            })
            raw_p.append(res["p_value"])
            keys.append((m1, m2))

    if raw_p:
        adj = holm_bonferroni(raw_p)
        for r, pa in zip(rows, adj):
            r["p_value_adj"]      = pa
            r["significant_0.05"] = pa < 0.05

    return pd.DataFrame(rows)


# ── Additional statistical tests ──────────────────────────────────────────────

def shapiro_wilk(sample: np.ndarray) -> Dict[str, float]:
    """Shapiro-Wilk test for normality. Null hyp: sample is normally distributed."""
    sample = np.asarray(sample)
    if len(sample) < 3:
        return {"statistic": float("nan"), "p_value": 1.0, "n": len(sample)}
    try:
        stat, p = stats.shapiro(sample)
        return {"statistic": float(stat), "p_value": float(p), "n": len(sample)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "n": len(sample)}


def levene_test(*samples: np.ndarray) -> Dict[str, float]:
    """Levene's test for homoscedasticity (equal variances). Null hyp: equal variances."""
    clean_samples = [np.asarray(s) for s in samples if len(s) > 0]
    if len(clean_samples) < 2:
        return {"statistic": float("nan"), "p_value": 1.0, "k": len(clean_samples)}
    try:
        stat, p = stats.levene(*clean_samples)
        return {"statistic": float(stat), "p_value": float(p), "k": len(clean_samples)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "k": len(clean_samples)}


def paired_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired Student's t-test. Parametric comparison of two paired groups."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(a) != len(b):
        return {"statistic": float("nan"), "p_value": 1.0, "n": len(a)}
    try:
        stat, p = stats.ttest_rel(a, b, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p), "n": len(a)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "n": len(a)}


def independent_ttest(a: np.ndarray, b: np.ndarray, equal_var: bool = False) -> Dict[str, float]:
    """Independent Student's t-test (default: Welch's t-test for unequal variances)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return {"statistic": float("nan"), "p_value": 1.0, "n_a": len(a), "n_b": len(b)}
    try:
        stat, p = stats.ttest_ind(a, b, equal_var=equal_var, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p), "n_a": len(a), "n_b": len(b)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "n_a": len(a), "n_b": len(b)}


def one_way_anova(*samples: np.ndarray) -> Dict[str, float]:
    """One-way ANOVA (analysis of variance). Null hyp: all groups have the same mean."""
    clean_samples = [np.asarray(s) for s in samples if len(s) > 1]
    if len(clean_samples) < 2:
        return {"statistic": float("nan"), "p_value": 1.0, "k": len(clean_samples)}
    try:
        stat, p = stats.f_oneway(*clean_samples)
        return {"statistic": float(stat), "p_value": float(p), "k": len(clean_samples)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": 1.0, "k": len(clean_samples)}


def tukey_hsd_test(*samples: np.ndarray, labels: List[str]) -> pd.DataFrame:
    """Tukey's Honestly Significant Difference (HSD) post-hoc test.
    Returns a dataframe of pairwise comparisons with adjusted p-values.
    """
    clean_samples = [np.asarray(s) for s in samples]
    if len(clean_samples) < 2 or len(clean_samples) != len(labels):
        return pd.DataFrame()
    
    # Check if stats.tukey_hsd is available (scipy >= 1.8.0)
    if hasattr(stats, "tukey_hsd"):
        try:
            res = stats.tukey_hsd(*clean_samples)
            rows = []
            n = len(labels)
            p_matrix = res.pvalue
            statistic_matrix = res.statistic
            for i in range(n):
                for j in range(i + 1, n):
                    diff = float(clean_samples[i].mean() - clean_samples[j].mean())
                    rows.append({
                        "group_a": labels[i],
                        "group_b": labels[j],
                        "mean_a": float(clean_samples[i].mean()),
                        "mean_b": float(clean_samples[j].mean()),
                        "diff": diff,
                        "statistic": float(statistic_matrix[i, j]),
                        "p_value": float(p_matrix[i, j]),
                        "significant_0.05": p_matrix[i, j] < 0.05
                    })
            return pd.DataFrame(rows)
        except Exception:
            pass

    # Fallback if tukey_hsd fails or is not present: perform pairwise independent t-tests with Holm correction
    rows = []
    n = len(labels)
    raw_p = []
    for i in range(n):
        for j in range(i + 1, n):
            res_t = independent_ttest(clean_samples[i], clean_samples[j], equal_var=False)
            diff = float(clean_samples[i].mean() - clean_samples[j].mean())
            rows.append({
                "group_a": labels[i],
                "group_b": labels[j],
                "mean_a": float(clean_samples[i].mean()),
                "mean_b": float(clean_samples[j].mean()),
                "diff": diff,
                "statistic": res_t["statistic"],
                "p_value": res_t["p_value"]
            })
            raw_p.append(res_t["p_value"])
            
    if raw_p:
        adj = holm_bonferroni(raw_p)
        for r, pa in zip(rows, adj):
            r["p_value"] = pa
            r["significant_0.05"] = pa < 0.05
            
    return pd.DataFrame(rows)

