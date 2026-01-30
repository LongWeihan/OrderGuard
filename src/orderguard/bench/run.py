from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from orderguard.modeling import load_lm
from orderguard.methods import latin_consensus, perm_consensus, single_shot_letter_scoring
from orderguard.tasks import TASKS, load_task


@dataclass(frozen=True)
class PerExampleRow:
    task: str
    model: str
    method: str
    idx: int
    gold: int
    pred: int | None
    correct: bool
    gold_prob: float | None
    top_prob: float | None
    entropy: float | None
    margin: float | None
    nll: float | None
    brier: float | None
    wall_s: float
    perms_used: int | None


def _model_tag(model_id: str) -> str:
    name = model_id.split("/", 1)[-1]
    return (
        name.replace("Instruct", "")
        .replace("-", "_")
        .replace(".", "_")
        .strip("_")
        .lower()
    )


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _tqdm(it, *, desc: str):
    # Some Windows shells / non-TTY pipes error on tqdm's carriage-return updates.
    # Default to disabled when not attached to a TTY.
    return tqdm(
        it,
        desc=desc,
        ascii=True,
        dynamic_ncols=True,
        disable=(not sys.stderr.isatty()),
    )


def _load_done(summary_path: Path) -> set[tuple[str, str, str]]:
    if not summary_path.exists():
        return set()
    done: set[tuple[str, str, str]] = set()
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = row.get("model")
            t = row.get("task")
            md = row.get("method")
            if isinstance(m, str) and isinstance(t, str) and isinstance(md, str):
                done.add((m, t, md))
    return done


def _bootstrap_ci(accs: list[int], *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    xs = np.array(accs, dtype=np.float32)
    n = xs.shape[0]
    if n == 0:
        return 0.0, 0.0
    boots = []
    for _ in range(n_boot):
        sample = xs[rng.integers(0, n, size=n)]
        boots.append(float(sample.mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975]).tolist()
    return float(lo), float(hi)


def _entropy(probs: list[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs)


def _margin(probs: list[float]) -> float:
    if len(probs) < 2:
        return 0.0
    xs = sorted(probs)
    return float(xs[-1] - xs[-2])


def _brier(probs: list[float], gold: int) -> float:
    out = 0.0
    for i, p in enumerate(probs):
        y = 1.0 if i == gold else 0.0
        out += (p - y) ** 2
    return float(out)


def _ece(confs: np.ndarray, correct: np.ndarray, *, n_bins: int = 15) -> float:
    # Expected calibration error over top-1 confidence.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:], strict=True):
        # last bin is inclusive on the right
        if b1 >= 1.0:
            m = (confs >= b0) & (confs <= b1)
        else:
            m = (confs >= b0) & (confs < b1)
        if not np.any(m):
            continue
        acc = float(np.mean(correct[m]))
        conf = float(np.mean(confs[m]))
        ece += float(np.mean(m)) * abs(acc - conf)
    return float(ece)


def run_benchmark(
    *,
    models: list[str],
    tasks: list[str],
    methods: list[str],
    out_dir: str,
    run_dir: str | None,
    resume: bool,
    max_examples: int,
    seed: int,
    max_perms: int,
    min_perms: int,
    js_eps: float,
    n_boot: int,
    torch_dtype: str | None,
    pad_to_multiple_of: int,
    attn_implementation: str | None,
    safe_mode: bool,
    warn_slow_example_s: float,
) -> Path:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    # "Progress freezes" on Windows are often long kernel compilation / attention backend issues.
    # Safe mode forces eager attention + pads to coarse buckets to reduce shape churn.
    if safe_mode:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    out_root = Path(out_dir)
    if run_dir:
        rd = Path(run_dir)
        run_dir_path = rd if rd.is_absolute() else (out_root / rd)
    else:
        run_dir_path = out_root / _utc_stamp()
    _ensure_dir(run_dir_path)

    summary_path = run_dir_path / "summary.jsonl"
    done = _load_done(summary_path) if resume else set()
    if summary_path.exists() and not resume:
        summary_path.unlink()
        done = set()

    state: dict[str, Any] = {"model": None, "task": None, "method": None, "idx": None}
    try:
        for model_id in models:
            state["model"] = model_id
            dtype = None if torch_dtype is None else getattr(torch, torch_dtype)
            lm = load_lm(
                model_id,
                torch_dtype=dtype,
                pad_to_multiple_of=pad_to_multiple_of,
                attn_implementation=attn_implementation,
            )

            for task_name in tasks:
                state["task"] = task_name
                spec = TASKS[task_name]
                ds = load_task(task_name)
                limit = max_examples if max_examples and max_examples > 0 else spec.default_limit
                if limit and limit > 0:
                    ds = ds.select(range(min(limit, len(ds))))

                for method in methods:
                    state["method"] = method
                    state["idx"] = None
                    if resume and (model_id, task_name, method) in done:
                        continue

                    t0 = time.perf_counter()
                    corrects: list[int] = []
                    per_example: list[PerExampleRow] = []

                    for i, ex in enumerate(_tqdm(ds, desc=f"{model_id}:{task_name}:{method}")):
                        state["idx"] = i
                        q, choices, gold = spec.to_mc(ex)
                        ex_t0 = time.perf_counter()

                        if method == "single":
                            r = single_shot_letter_scoring(lm, q, choices, seed=seed + i)
                            perms_used = None
                        elif method == "pcons":
                            r = perm_consensus(
                                lm,
                                q,
                                choices,
                                max_perms=max_perms,
                                min_perms=min_perms,
                                js_eps=js_eps,
                                seed=seed + i,
                            )
                            perms_used = int(r.meta.get("perms_used", 0)) if r.meta else None
                        elif method == "latin":
                            r = latin_consensus(
                                lm,
                                q,
                                choices,
                                max_perms=max_perms,
                                min_perms=min_perms,
                                js_eps=js_eps,
                                seed=seed + i,
                            )
                            perms_used = int(r.meta.get("perms_used", 0)) if r.meta else None
                        else:
                            raise RuntimeError(method)

                        wall = time.perf_counter() - ex_t0
                        if warn_slow_example_s and wall > warn_slow_example_s:
                            warnings.warn(
                                f"Slow example: model={model_id} task={task_name} method={method} idx={i} wall_s={wall:.2f}",
                                RuntimeWarning,
                                stacklevel=1,
                            )
                        pred = r.pred_index
                        corr = (pred == gold)
                        corrects.append(1 if corr else 0)
                        gold_prob = None
                        top_prob = None
                        entropy = None
                        margin = None
                        nll = None
                        brier = None
                        if r.probs is not None and len(r.probs) > gold:
                            probs = [float(p) for p in r.probs]
                            gold_prob = float(probs[gold])
                            top_prob = float(max(probs))
                            entropy = _entropy(probs)
                            margin = _margin(probs)
                            nll = float(-math.log(max(gold_prob, 1e-12)))
                            brier = _brier(probs, gold)
                        per_example.append(
                            PerExampleRow(
                                task=task_name,
                                model=model_id,
                                method=method,
                                idx=i,
                                gold=gold,
                                pred=pred,
                                correct=corr,
                                gold_prob=gold_prob,
                                top_prob=top_prob,
                                entropy=entropy,
                                margin=margin,
                                nll=nll,
                                brier=brier,
                                wall_s=wall,
                                perms_used=perms_used,
                            )
                        )

                    wall_total = time.perf_counter() - t0
                    df = pd.DataFrame([asdict(r) for r in per_example])

                    out_jsonl = (
                        run_dir_path / f"per_example__{task_name}__{method}__{model_id.replace('/', '__')}.jsonl"
                    )
                    with out_jsonl.open("w", encoding="utf-8") as f:
                        for row in per_example:
                            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

                    acc = float(np.mean(corrects)) if corrects else 0.0
                    lo, hi = _bootstrap_ci(corrects, n_boot=n_boot, seed=seed)
                    avg_perms = None
                    if "perms_used" in df:
                        s = df["perms_used"].dropna()
                        avg_perms = float(s.mean()) if len(s) else None

                    mean_nll = float(df["nll"].mean()) if "nll" in df and df["nll"].notna().any() else None
                    mean_brier = float(df["brier"].mean()) if "brier" in df and df["brier"].notna().any() else None
                    mean_entropy = float(df["entropy"].mean()) if "entropy" in df and df["entropy"].notna().any() else None
                    mean_margin = float(df["margin"].mean()) if "margin" in df and df["margin"].notna().any() else None
                    mean_top_prob = (
                        float(df["top_prob"].mean()) if "top_prob" in df and df["top_prob"].notna().any() else None
                    )
                    ece15 = None
                    if "top_prob" in df and df["top_prob"].notna().any():
                        confs = df["top_prob"].fillna(0.0).to_numpy(dtype=np.float64)
                        corr_np = df["correct"].to_numpy(dtype=np.float64)
                        ece15 = _ece(confs, corr_np, n_bins=15)

                    summary_row: dict[str, Any] = {
                        "task": task_name,
                        "model": model_id,
                        "model_tag": _model_tag(model_id),
                        "method": method,
                        "n": int(len(corrects)),
                        "accuracy": acc,
                        "acc_ci95_lo": lo,
                        "acc_ci95_hi": hi,
                        "mean_nll": mean_nll,
                        "mean_brier": mean_brier,
                        "ece15": ece15,
                        "mean_entropy": mean_entropy,
                        "mean_margin": mean_margin,
                        "mean_top_prob": mean_top_prob,
                        "wall_s_total": wall_total,
                        "avg_s_per_example": float(df["wall_s"].mean()) if len(df) else 0.0,
                        "avg_perms_used": avg_perms,
                        "max_perms": max_perms if method in {"pcons", "latin"} else None,
                        "min_perms": min_perms if method in {"pcons", "latin"} else None,
                        "js_eps": js_eps if method in {"pcons", "latin"} else None,
                        "torch_dtype": lm.torch_dtype,
                    }
                    with summary_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(summary_row, ensure_ascii=False) + "\n")
                    done.add((model_id, task_name, method))
    except BaseException:
        err_path = run_dir_path / "error.log"
        payload = {
            "state": state,
            "traceback": traceback.format_exc(),
        }
        err_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise

    summary_df = pd.read_json(summary_path, lines=True)
    summary_df.to_csv(run_dir_path / "summary.csv", index=False)
    return run_dir_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume an existing run_dir by skipping completed (model,task,method) rows in summary.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--torch_dtype", default=None, choices=[None, "float16", "bfloat16"])
    parser.add_argument("--max_perms", type=int, default=7)
    parser.add_argument("--min_perms", type=int, default=3)
    parser.add_argument("--js_eps", type=float, default=0.005)
    parser.add_argument("--pad_to_multiple_of", type=int, default=32)
    parser.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa"])
    parser.add_argument("--warn_slow_example_s", type=float, default=10.0)
    parser.add_argument(
        "--safe_mode",
        action=argparse.BooleanOptionalAction,
        default=(os.name == "nt"),
        help="More stable (esp. on Windows): disables Flash/ME SDPA and prefers eager attention.",
    )
    parser.add_argument("--methods", nargs="+", default=["single", "pcons", "latin"], choices=["single", "pcons", "latin"])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "arc_challenge",
            "openbookqa",
            "commonsenseqa",
            "truthfulqa_mc1",
            "mmlu_all",
            "hellaswag",
            "winogrande_xs",
        ],
        choices=sorted(TASKS.keys()),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"],
    )
    args = parser.parse_args()

    attn_impl = args.attn_implementation
    if args.safe_mode and attn_impl is None:
        attn_impl = "eager"

    run_dir = run_benchmark(
        models=args.models,
        tasks=args.tasks,
        methods=args.methods,
        out_dir=args.out_dir,
        run_dir=args.run_dir,
        resume=bool(args.resume),
        max_examples=args.max_examples,
        seed=args.seed,
        max_perms=args.max_perms,
        min_perms=args.min_perms,
        js_eps=args.js_eps,
        n_boot=args.n_boot,
        torch_dtype=args.torch_dtype,
        pad_to_multiple_of=args.pad_to_multiple_of,
        attn_implementation=attn_impl,
        safe_mode=bool(args.safe_mode),
        warn_slow_example_s=args.warn_slow_example_s,
    )
    latest_dir = Path(args.out_dir) / "latest"
    _ensure_dir(latest_dir)
    for p in latest_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in {"summary.csv", "summary.jsonl"} or p.name.startswith("per_example__"):
            p.unlink()
    for p in run_dir.iterdir():
        if p.is_file():
            (latest_dir / p.name).write_bytes(p.read_bytes())

    print(f"Saved run to: {run_dir}")
    print(f"Updated latest to: {latest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
