from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_LABELS = {
    "arc_challenge": "ARC-C",
    "openbookqa": "OpenBookQA",
    "commonsenseqa": "CSQA",
    "truthfulqa_mc1": "TruthfulQA (MC1)",
    "mmlu_all": "MMLU (all)",
    "hellaswag": "HellaSwag",
    "winogrande_xs": "WinoGrande-XS",
    "boolq": "BoolQ",
}

METHOD_LABELS = {
    "single": "Single",
    "pcons": "PCons (random)",
    "latin": "LaTIn (Latin)",
}

METHOD_COLORS = {
    "single": "#4C78A8",
    "pcons": "#F58518",
    "latin": "#54A24B",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _nice_task(t: str) -> str:
    return TASK_LABELS.get(t, t)


def _nice_method(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def _per_example_path(run_dir: Path, *, task: str, method: str, model_id: str) -> Path:
    return run_dir / f"per_example__{task}__{method}__{model_id.replace('/', '__')}.jsonl"


def _load_correct(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return np.zeros((0,), dtype=np.int8)
    rows.sort(key=lambda r: int(r["idx"]))
    return np.array([1 if r["correct"] else 0 for r in rows], dtype=np.int8)


def _paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Paired bootstrap CI for mean(a - b). a and b are 0/1 arrays aligned by example.
    Returns (delta_mean, ci_lo, ci_hi).
    """
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    d = a.astype(np.float32) - b.astype(np.float32)
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975]).tolist()
    return delta, float(lo), float(hi)


def _save(fig, out_png: Path, out_svg: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def _fig_accuracy_gain(
    run_dir: Path,
    df: pd.DataFrame,
    *,
    out_dir: Path,
    tasks_order: list[str],
    models_order: list[str],
    methods: list[str],
    n_boot: int,
    seed: int,
) -> None:
    # Build paired deltas + CIs from per-example logs.
    rows = []
    for model_id in models_order:
        for task in tasks_order:
            p_single = _per_example_path(run_dir, task=task, method="single", model_id=model_id)
            if not p_single.exists():
                continue
            base = _load_correct(p_single)
            for method in methods:
                p_m = _per_example_path(run_dir, task=task, method=method, model_id=model_id)
                if not p_m.exists():
                    continue
                cur = _load_correct(p_m)
                delta, lo, hi = _paired_bootstrap_delta(cur, base, n_boot=n_boot, seed=seed)
                rows.append(
                    {
                        "model": model_id,
                        "task": task,
                        "method": method,
                        "delta": delta,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    ddf = pd.DataFrame(rows)
    if len(ddf) == 0:
        return

    # Plot (horizontal, per model)
    _set_style()
    model_titles = {
        "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
        "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    }
    fig, axes = plt.subplots(
        nrows=len(models_order),
        ncols=1,
        figsize=(9.5, 3.4 * len(models_order)),
        constrained_layout=True,
    )
    if len(models_order) == 1:
        axes = [axes]

    for ax, model_id in zip(axes, models_order, strict=True):
        sub = ddf[ddf["model"] == model_id].copy()
        if len(sub) == 0:
            ax.axis("off")
            continue

        # Ensure task ordering.
        sub["task"] = pd.Categorical(sub["task"], categories=tasks_order, ordered=True)
        sub = sub.sort_values(["task", "method"])

        y0 = np.arange(len(tasks_order), dtype=np.float32)
        bar_h = 0.32
        offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods)) * bar_h

        for off, method in zip(offsets, methods, strict=True):
            s = sub[sub["method"] == method].set_index("task")
            xs = []
            xerr_lo = []
            xerr_hi = []
            for t in tasks_order:
                if t in s.index:
                    r = s.loc[t]
                    xs.append(float(r["delta"]) * 100.0)
                    xerr_lo.append((float(r["delta"]) - float(r["ci_lo"])) * 100.0)
                    xerr_hi.append((float(r["ci_hi"]) - float(r["delta"])) * 100.0)
                else:
                    xs.append(np.nan)
                    xerr_lo.append(np.nan)
                    xerr_hi.append(np.nan)

            xs = np.array(xs, dtype=np.float64)
            xerr = np.vstack([np.array(xerr_lo), np.array(xerr_hi)])
            msk = np.isfinite(xs)
            ax.barh(
                y0[msk] + off,
                xs[msk],
                height=bar_h,
                color=METHOD_COLORS.get(method, "#999999"),
                label=_nice_method(method),
                xerr=xerr[:, msk],
                error_kw={"ecolor": "black", "elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
            )

        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(y0)
        ax.set_yticklabels([_nice_task(t) for t in tasks_order])
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy gain vs single (percentage points)")
        ax.set_title(f"{model_titles.get(model_id, model_id)}: accuracy gains (paired bootstrap 95% CI)")
        ax.legend(loc="lower right", frameon=True)

    _save(fig, out_dir / "accuracy_gain_by_task.png", out_dir / "accuracy_gain_by_task.svg")


def _fig_macro_accuracy(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    tasks_order: list[str],
    models_order: list[str],
    methods_order: list[str],
) -> None:
    _set_style()
    sub = df[df["task"].isin(tasks_order) & df["model"].isin(models_order) & df["method"].isin(methods_order)].copy()
    if len(sub) == 0:
        return

    macro = sub.pivot_table(index="model", columns="method", values="accuracy", aggfunc="mean").reindex(models_order)
    fig, ax = plt.subplots(figsize=(8.6, 3.6), constrained_layout=True)

    x = np.arange(len(models_order), dtype=np.float32)
    width = 0.24
    offsets = np.linspace(-(len(methods_order) - 1) / 2, (len(methods_order) - 1) / 2, len(methods_order)) * width

    for off, method in zip(offsets, methods_order, strict=True):
        ys = (macro[method] * 100.0).to_numpy(dtype=np.float64)
        bars = ax.bar(
            x + off,
            ys,
            width=width,
            label=_nice_method(method),
            color=METHOD_COLORS.get(method, "#999999"),
        )
        for b, y in zip(bars, ys, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                y + 0.6,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Annotate deltas above pcons/latin bars.
    for i, model_id in enumerate(models_order):
        base = float(macro.loc[model_id, "single"] * 100.0)
        for method, off in zip(["pcons", "latin"], offsets[1:], strict=True):
            if method not in macro.columns:
                continue
            y = float(macro.loc[model_id, method] * 100.0)
            ax.text(
                x[i] + off,
                y + 3.0,
                f"{(y - base):+0.1f} pp",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Qwen3-0.6B", "Qwen3-1.7B"])
    ax.set_ylabel("Macro accuracy across 7 datasets (%)")
    ax.set_ylim(0, 100)
    ax.set_title("OrderGuard improves accuracy (macro-average)")
    ax.legend(loc="lower right", frameon=True)

    _save(fig, out_dir / "macro_accuracy.png", out_dir / "macro_accuracy.svg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    df = pd.read_csv(run_dir / "summary.csv")

    tasks_order = [
        "arc_challenge",
        "openbookqa",
        "commonsenseqa",
        "truthfulqa_mc1",
        "mmlu_all",
        "hellaswag",
        "winogrande_xs",
    ]
    models_order = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
    methods_order = ["single", "pcons", "latin"]

    figs_dir = run_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Core figure: per-task accuracy gains (with correctly aligned error bars)
    _fig_accuracy_gain(
        run_dir,
        df,
        out_dir=figs_dir,
        tasks_order=tasks_order,
        models_order=models_order,
        methods=["pcons", "latin"],
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # 2) Macro accuracy summary
    _fig_macro_accuracy(
        df,
        out_dir=figs_dir,
        tasks_order=tasks_order,
        models_order=models_order,
        methods_order=methods_order,
    )

    print(f"Wrote figures to: {figs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
