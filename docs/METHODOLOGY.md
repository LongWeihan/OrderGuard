# Methodology

OrderGuard targets a common failure mode in LLM-based multiple-choice judging and candidate ranking:
**the predicted winner can change when the same options are presented in a different order**.

This repo implements two order-robust test-time wrappers:

## 1) `pcons` (Permutation Consensus, random)

Treat option order as a nuisance variable. For each trial:

1. Randomly permute options.
2. Score answer letters `A/B/C/...` by **log-probability** (no free-form generation).
3. Map letter scores back to original options.
4. Aggregate across trials (sum logprobs).
5. Stop early when the aggregated distribution stabilizes (JS divergence < `js_eps`).

## 2) `latin` (Permutation Consensus, Latin-square / cyclic design)

`latin` is a lower-variance, more compute-efficient variant of `pcons`:

1. Sample one random **base permutation**.
2. Evaluate **cyclic shifts** of that base permutation.
3. With `n` shifts for `n` options, each option appears in each position exactly once.

This targets position bias directly and often reaches the same accuracy with fewer trials.

## Early stopping

Both methods maintain an aggregated score vector and convert it to a probability distribution via softmax.
If the Jensen-Shannon divergence between successive distributions is below `js_eps` for at least `min_perms`
trials, the method stops (saving compute on easy examples).
