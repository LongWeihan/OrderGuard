from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from orderguard.modeling import batch_logprob_completions, load_lm
from orderguard.prompting import make_mc_messages
from orderguard.tasks import TASKS, load_task


@dataclass(frozen=True)
class SensRow:
    task: str
    model: str
    model_tag: str
    n_examples: int
    n_perms: int
    flip_rate: float
    avg_unique_preds: float
    wall_s_total: float
    avg_s_per_example: float


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _tqdm(it, *, desc: str):
    return tqdm(
        it,
        desc=desc,
        ascii=True,
        dynamic_ncols=True,
        disable=(not sys.stderr.isatty()),
    )


def _single_pred_index(model, question: str, choices: list[str], *, seed: int) -> int:
    n = len(choices)
    letters = [chr(ord("A") + i) for i in range(n)]
    messages = make_mc_messages(question, choices, label_letters=letters)
    lps = batch_logprob_completions(model, messages, letters)
    return int(max(range(n), key=lambda i: lps[i]))


def _model_tag(model_id: str) -> str:
    name = model_id.split("/", 1)[-1]
    return (
        name.replace("Instruct", "")
        .replace("-", "_")
        .replace(".", "_")
        .strip("_")
        .lower()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--perms", type=int, default=10)
    parser.add_argument("--pad_to_multiple_of", type=int, default=32)
    parser.add_argument("--attn_implementation", default=None, choices=["eager", "sdpa"])
    parser.add_argument(
        "--safe_mode",
        action=argparse.BooleanOptionalAction,
        default=(os.name == "nt"),
        help="More stable (esp. on Windows): disables Flash/ME SDPA and prefers eager attention.",
    )
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

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.safe_mode:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    out_root = Path(args.out_dir)
    run_dir = out_root / f"sensitivity_{_utc_stamp()}"
    latest_dir = out_root / "latest"
    _ensure_dir(run_dir)
    _ensure_dir(latest_dir)

    rows: list[SensRow] = []

    attn_impl = args.attn_implementation
    if args.safe_mode and attn_impl is None:
        attn_impl = "eager"

    for model_id in args.models:
        lm = load_lm(
            model_id,
            torch_dtype=None,
            pad_to_multiple_of=args.pad_to_multiple_of,
            attn_implementation=attn_impl,
        )
        for task_name in args.tasks:
            spec = TASKS[task_name]
            ds = load_task(task_name)
            n = min(args.examples, len(ds))
            ds = ds.select(range(n))

            flips = 0
            unique_counts: list[int] = []
            t0 = time.perf_counter()
            g = torch.Generator(device="cpu")
            g.manual_seed(args.seed)

            for i, ex in enumerate(_tqdm(ds, desc=f"sens:{model_id}:{task_name}")):
                q, choices, _gold = spec.to_mc(ex)
                base_pred = _single_pred_index(lm, q, choices, seed=args.seed + i)

                preds = {base_pred}
                flipped = False
                for j in range(args.perms):
                    perm = torch.randperm(len(choices), generator=g).tolist()
                    perm_choices = [choices[k] for k in perm]
                    perm_pred_pos = _single_pred_index(lm, q, perm_choices, seed=args.seed + i * 10 + j)
                    perm_pred_orig = perm[perm_pred_pos]
                    preds.add(perm_pred_orig)
                    if perm_pred_orig != base_pred:
                        flipped = True
                if flipped:
                    flips += 1
                unique_counts.append(len(preds))

            wall = time.perf_counter() - t0
            row = SensRow(
                task=task_name,
                model=model_id,
                model_tag=_model_tag(model_id),
                n_examples=n,
                n_perms=args.perms,
                flip_rate=float(flips / n) if n else 0.0,
                avg_unique_preds=float(np.mean(unique_counts)) if unique_counts else 0.0,
                wall_s_total=wall,
                avg_s_per_example=float(wall / n) if n else 0.0,
            )
            rows.append(row)

    out_csv = run_dir / "sensitivity.csv"
    pd.DataFrame([asdict(r) for r in rows]).to_csv(out_csv, index=False)

    # Mirror into latest
    (latest_dir / "sensitivity.csv").write_bytes(out_csv.read_bytes())

    print(f"Wrote: {out_csv}")
    print(f"Updated: {latest_dir / 'sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
