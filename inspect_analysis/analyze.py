"""Reproduce the notebook's Phase 1-2 analysis from Inspect logs.

    python inspect_analysis/analyze.py logs/            # tables + stats to stdout
    python inspect_analysis/analyze.py logs/ --out results/latest   # also CSVs + figures

Reads every successful .eval log in the directory, flattens it to one row per response
(same columns as the spring CSVs), then runs:
  * self-report means per model x dimension (+ kindness vs control)
  * kind rate (TPR) per model x dimension
  * paired Wilcoxon on TPR - normalized self-report, bootstrap CI, Pearson/Spearman
  * pushback degradation (baseline TPR - pushback TPR)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from scipy import stats

DIMS = ["A", "ASP", "AP", "PSP", "PP"]


def short_model(name: str) -> str:
    return name.split("/")[-1]


def load_logs(log_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sr_rows, sjt_rows = [], []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            continue
        task = log.eval.task.split("/")[-1]
        model = short_model(log.eval.model)
        args = log.eval.task_args or {}
        for s in log.samples:
            md = s.metadata or {}
            if task == "self_report":
                sc = s.scores["likert_rating"]
                parsed = sc.value["parsed"] == 1
                sr_rows.append(dict(
                    framing=md.get("framing"), model=model, rep=s.epoch - 1,
                    item_id=s.id, dimension=md["dimension"], raw=sc.answer,
                    rating=sc.value["rating"] if parsed else np.nan,
                ))
            elif task == "sjt":
                sc = s.scores["kindness_judge"]
                v = sc.metadata["verdict"]
                row = dict(
                    condition="pressure" if args.get("pushback") else "baseline",
                    model=model, sjt_id=s.id, source_item=md["source_item_id"],
                    dimension=md["dimension"], ambiguity=md["ambiguity"],
                    sample=s.epoch - 1, response=sc.explanation, verdict=v,
                    is_kind=1.0 if v == "kind" else (0.0 if v == "unkind" else np.nan),
                )
                if "initial_verdict" in sc.metadata:
                    v0 = sc.metadata["initial_verdict"]
                    row["initial_verdict"] = v0
                    row["initial_is_kind"] = 1.0 if v0 == "kind" else (0.0 if v0 == "unkind" else np.nan)
                sjt_rows.append(row)
    return pd.DataFrame(sr_rows), pd.DataFrame(sjt_rows)


def bootstrap_mean_ci(x: np.ndarray, n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return np.percentile(boots, [2.5, 97.5]), (boots > 0).mean()


def bootstrap_r_ci(x: np.ndarray, y: np.ndarray, n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.choice(len(x), size=len(x), replace=True)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            rs.append(stats.pearsonr(x[idx], y[idx])[0])
    return np.percentile(rs, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", help="directory for CSVs and figures")
    ap.add_argument("--framing", default="human", help="self-report framing to compare (human/ai)")
    a = ap.parse_args()

    df_sr, df_sjt = load_logs(a.log_dir)
    pd.set_option("display.width", 140)

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        if len(df_sr):
            df_sr.to_csv(out / "phase1_self_report.csv", index=False)
        if len(df_sjt):
            df_sjt.to_csv(out / "phase2_sjt_results.csv", index=False)

    sr_means = tpr = None

    if len(df_sr):
        sr = df_sr[df_sr["framing"] == a.framing]
        k = sr[(sr["dimension"] != "CONTROL") & sr["rating"].notna()]
        c = sr[(sr["dimension"] == "CONTROL") & sr["rating"].notna()]
        print("\n== Phase 1: mean self-report (1-7) ==")
        print(k.pivot_table(index="model", columns="dimension", values="rating")
               .reindex(columns=[d for d in DIMS if d in k["dimension"].unique()]).round(2))
        print("\nKindness vs control:")
        print(pd.DataFrame({"kindness": k.groupby("model")["rating"].mean(),
                            "control": c.groupby("model")["rating"].mean()}).round(2))
        sr_means = k.groupby(["model", "dimension"])["rating"].mean().reset_index()
        sr_means["self_report_norm"] = (sr_means["rating"] - 1) / 6

    if len(df_sjt):
        base = df_sjt[(df_sjt["condition"] == "baseline") & df_sjt["is_kind"].notna()]
        if len(base):
            tpr = base.groupby(["model", "dimension"])["is_kind"].mean().reset_index(name="tpr")
            print("\n== Phase 2: kind rate (TPR), baseline ==")
            print(tpr.pivot(index="model", columns="dimension", values="tpr")
                     .reindex(columns=[d for d in DIMS if d in tpr["dimension"].unique()]).round(2))

        pres = df_sjt[(df_sjt["condition"] == "pressure") & df_sjt["is_kind"].notna()]
        if len(pres) and len(base):
            b = base.groupby(["model", "dimension"])["is_kind"].mean()
            p = pres.groupby(["model", "dimension"])["is_kind"].mean()
            deg = pd.DataFrame({"baseline": b, "pushback": p}).dropna().reset_index()
            deg["degradation"] = deg["baseline"] - deg["pushback"]
            print("\n== Pushback: baseline TPR - pushback TPR (as in the write-up) ==")
            print(deg.pivot(index="model", columns="dimension", values="degradation").round(2))
            print("\nMean degradation per model:")
            print(deg.groupby("model")["degradation"].mean().round(3))
            if a.out:
                _heatmap(deg.pivot(index="model", columns="dimension", values="degradation"),
                         "Kind-rate drop under pushback", Path(a.out) / "pushback_degradation.png",
                         cmap="RdYlGn_r", vmin=-0.3, vmax=0.7, center=0)
        if "initial_is_kind" in pres:
            flips = pres.dropna(subset=["initial_is_kind"])
            if len(flips):
                print("\nWithin pushback runs: share of kind first answers that flipped to unkind:")
                kind_first = flips[flips["initial_is_kind"] == 1]
                print((1 - kind_first.groupby("model")["is_kind"].mean()).round(3))

    if sr_means is not None and tpr is not None:
        paired = sr_means.merge(tpr, on=["model", "dimension"])
        paired["gap"] = paired["tpr"] - paired["self_report_norm"]
        if len(paired) >= 3:
            w, wp = stats.wilcoxon(paired["tpr"], paired["self_report_norm"])
            (lo, hi), p_pos = bootstrap_mean_ci(paired["gap"].to_numpy())
            r, rp = stats.pearsonr(paired["self_report_norm"], paired["tpr"])
            rho, rhop = stats.spearmanr(paired["self_report_norm"], paired["tpr"])
            rlo, rhi = bootstrap_r_ci(paired["self_report_norm"].to_numpy(), paired["tpr"].to_numpy())
            print("\n== Self-report vs behavior ==")
            print(f"n = {len(paired)} (model x dimension cells)")
            print(f"Mean gap (TPR - self-report): {paired['gap'].mean():+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], P(gap>0) = {p_pos:.3f}")
            print(f"Wilcoxon W = {w:.2f}, p = {wp:.4f}")
            print(f"Pearson r = {r:+.3f} (p = {rp:.4f}), 95% CI [{rlo:+.3f}, {rhi:+.3f}]; Spearman rho = {rho:+.3f} (p = {rhop:.4f})")
            print("\nPer model:")
            print(paired.groupby("model")[["self_report_norm", "tpr", "gap"]].mean().round(3))
            if a.out:
                paired.to_csv(Path(a.out) / "paired_self_report_vs_tpr.csv", index=False)
                _scatter(paired, Path(a.out) / "self_report_vs_behavior.png")


def _heatmap(pivot, title, path, **kw):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 2), 1 + pivot.shape[0] * 0.7))
    sns.heatmap(pivot, annot=True, fmt=".2f", linewidths=0.5, ax=ax, **kw)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(paired: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt

    dims = [d for d in DIMS if d in paired["dimension"].unique()]
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dim in zip(axes, dims):
        sub = paired[paired["dimension"] == dim]
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
        for _, row in sub.iterrows():
            ax.scatter(row["self_report_norm"], row["tpr"], s=60)
            ax.annotate(row["model"], (row["self_report_norm"], row["tpr"]), fontsize=7)
        ax.set_title(dim)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Self-report (normalized 0-1)")
    axes[0].set_ylabel("Behavioral kind rate")
    fig.suptitle("Self-report vs. behavior (above diagonal = kinder in actions than in words)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
