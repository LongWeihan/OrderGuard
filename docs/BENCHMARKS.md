# Benchmarks

OrderGuard provides two complementary benchmark suites:

## 1) Accuracy + cost + calibration (`bench.run`)

Evaluates multiple-choice accuracy and several practical metrics:

- `accuracy` (+ bootstrap 95% CI)
- latency: `avg_s_per_example`
- adaptive compute: `avg_perms_used` (for `pcons` / `latin`)
- calibration proxies: `mean_nll`, `mean_brier`, `ece15`

Run:

```powershell
cd orderguard
.\.venv\Scripts\python -m orderguard.bench.run --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B
.\.venv\Scripts\python -m orderguard.plots.make_figures --run_dir reports/latest
```

Paper artifact (already computed in this repo): `reports/paper_qwen3/`

## 2) Order sensitivity (`bench.sensitivity`)

Quantifies how unstable single-shot judging is under random option shuffles:

- `flip_rate`: fraction of examples where the predicted winner changes under permutations
- `avg_unique_preds`: how many distinct winners appear across shuffles

Run:

```powershell
cd orderguard
.\.venv\Scripts\python -m orderguard.bench.sensitivity --models Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --examples 200 --perms 10
```
