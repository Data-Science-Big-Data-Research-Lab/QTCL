"""Tables of the article computed from the run records (results/*/runs.csv).

    python report.py --results ./results --out ./tables

writes ablation_qk_ideal.tex (circuit ablation) and equivalence_tests.tex (paired TOST of the
MLP head against each hybrid head in the main benchmark). Every number comes from a run record:
a cell without runs is printed as "---" and a standard deviation is printed only with two or more
runs.
"""
import argparse
import glob
import os
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SEEDS_COLUMN = "seed"


def load_runs(results_dir: str) -> pd.DataFrame:
    """One row per (study, run_id). If a run_id was recorded more than once, the first record in
    folder-name order is kept."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*", "runs.csv")))
    if not paths:
        raise FileNotFoundError(f"no runs.csv under {results_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["study"] = df.get("study", pd.Series(["main"] * len(df))).fillna("main")
    return df.drop_duplicates(subset=["study", "run_id"], keep="first").reset_index(drop=True)


def mean_std(values: Iterable[float], scale: float = 100.0) -> str:
    """'$m \\pm s$' over the values (sample standard deviation), '$m$' for one value, '---' for none."""
    v = np.asarray([x for x in values if pd.notna(x)], dtype=float) * scale
    if v.size == 0:
        return "---"
    if v.size == 1:
        return f"${v[0]:.2f}$"
    return f"${v.mean():.2f} \\pm {v.std(ddof=1):.2f}$"


def paired_by_seed(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pair two sets of runs by seed. Raises ValueError if their seeds differ or repeat."""
    for name, d in (("a", a), ("b", b)):
        if d[SEEDS_COLUMN].duplicated().any():
            raise ValueError(f"repeated seeds in {name}: {sorted(d[SEEDS_COLUMN])}")
    sa, sb = set(a[SEEDS_COLUMN]), set(b[SEEDS_COLUMN])
    if sa != sb:
        raise ValueError(f"unpaired seeds: {sorted(sa ^ sb)}")
    return a[[SEEDS_COLUMN, metric]].merge(b[[SEEDS_COLUMN, metric]], on=SEEDS_COLUMN, suffixes=("_a", "_b"))


def tost_paired(d: np.ndarray, margin: float, alpha: float = 0.05) -> Tuple[float, Tuple[float, float], float]:
    """Two one-sided t tests on paired differences d for equivalence within [-margin, margin].
    Returns the mean difference, the (1 - 2 alpha) confidence interval and the TOST p-value."""
    d = np.asarray(d, dtype=float)
    n = d.size
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(1 - alpha, n - 1)
    p = max(stats.t.sf((d.mean() + margin) / se, n - 1), stats.t.cdf((d.mean() - margin) / se, n - 1))
    return float(d.mean()), (float(d.mean() - tc * se), float(d.mean() + tc * se)), float(p)


def ablation_table(df: pd.DataFrame, head: str = "qk_ideal") -> str:
    ab = df[(df["study"] == "ablation") & (df["head"] == head)]
    qubits = sorted(set(ab["n_qubits"].astype(int))) or [2, 4, 6, 8]
    depths = sorted(set(ab["depth"].astype(int))) or [1, 2, 3]
    lines = ["\\begin{tabular}{l" + "c" * len(depths) + "}", "\\toprule",
             "$n_q$ & " + " & ".join(f"$L = {d}$" for d in depths) + " \\\\", "\\midrule"]
    for q in qubits:
        cells = [mean_std(ab[(ab["n_qubits"].astype(int) == q) & (ab["depth"].astype(int) == d)]["AA"]) for d in depths]
        lines.append(f"{q} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def tost_table(df: pd.DataFrame, margin: float = 2.0) -> str:
    main = df[df["study"] == "main"]
    lines = ["\\begin{tabular}{lllcccc}", "\\toprule",
             "Dataset & Backbone & Comparison & Mean diff. (pp) & 90\\% CI & TOST $p$ & Equivalent \\\\", "\\midrule"]
    for (ds, bb), g in main.groupby(["dataset", "backbone"], sort=True):
        ref = g[g["head"] == "mlp"]
        for head in sorted(set(g["head"]) - {"mlp"}):
            pair = paired_by_seed(ref, g[g["head"] == head], "AA")
            m, (lo, hi), p = tost_paired(100 * (pair["AA_a"] - pair["AA_b"]).to_numpy(), margin)
            lines.append(f"{ds.replace('_', ' ')} & {bb.replace('_', chr(92) + '_')} & mlp vs {head.replace('_', chr(92) + '_')} & "
                         f"${m:.2f}$ & $[{lo:.2f}, {hi:.2f}]$ & ${p:.4f}$ & {'Yes' if p < 0.05 else 'No'} \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="./results")
    p.add_argument("--out", default="./tables")
    p.add_argument("--margin", type=float, default=2.0, help="TOST equivalence margin in points of AA")
    args = p.parse_args()
    df = load_runs(args.results)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation_qk_ideal.tex"), "w") as f:
        f.write(ablation_table(df))
    with open(os.path.join(args.out, "equivalence_tests.tex"), "w") as f:
        f.write(tost_table(df, args.margin))
    print(f"tables written to {args.out}")


if __name__ == "__main__":
    main()
