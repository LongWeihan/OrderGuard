from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_run_dir", required=True, help="e.g. reports/20260129_151636")
    parser.add_argument("--dst_run_dir", required=True, help="e.g. reports/paper_qwen3")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    src = Path(args.src_run_dir)
    dst = Path(args.dst_run_dir)
    _ensure_dir(dst)

    src_summary = src / "summary.jsonl"
    if not src_summary.exists():
        raise FileNotFoundError(src_summary)

    rows = _read_jsonl(src_summary)
    keep = [r for r in rows if r.get("model") in set(args.models)]

    out_jsonl = dst / "summary.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.read_json(out_jsonl, lines=True).to_csv(dst / "summary.csv", index=False)

    # Copy per-example logs for kept models (if present).
    for p in src.glob("per_example__*__*.jsonl"):
        name = p.name
        # filename contains model_id with '/' replaced by '__'
        if any(m.replace("/", "__") in name for m in args.models):
            (dst / name).write_bytes(p.read_bytes())

    # Copy sensitivity if present.
    sens = src / "sensitivity.csv"
    if sens.exists():
        (dst / "sensitivity.csv").write_bytes(sens.read_bytes())

    print(f"Wrote: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

