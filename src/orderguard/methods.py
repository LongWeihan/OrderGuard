from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from orderguard.modeling import LoadedModel, batch_logprob_completions
from orderguard.prompting import make_mc_messages


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _kl(p: list[float], q: list[float]) -> float:
    eps = 1e-12
    out = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi <= 0:
            continue
        out += pi * math.log(pi / max(qi, eps))
    return out


def _js(p: list[float], q: list[float]) -> float:
    m = [(pi + qi) / 2 for pi, qi in zip(p, q, strict=True)]
    return (_kl(p, m) + _kl(q, m)) / 2


@dataclass(frozen=True)
class McResult:
    pred_index: int | None
    probs: list[float] | None
    meta: dict[str, Any]


def single_shot_letter_scoring(
    model: LoadedModel,
    question: str,
    choices: list[str],
    *,
    seed: int,
) -> McResult:
    n = len(choices)
    letters = [chr(ord("A") + i) for i in range(n)]
    messages = make_mc_messages(question, choices, label_letters=letters)
    lps = batch_logprob_completions(model, messages, letters)
    pred = max(range(n), key=lambda i: lps[i])
    return McResult(pred_index=pred, probs=_softmax(lps), meta={"logprobs": lps, "seed": seed})


def perm_consensus(
    model: LoadedModel,
    question: str,
    choices: list[str],
    *,
    max_perms: int = 7,
    min_perms: int = 3,
    js_eps: float = 0.005,
    seed: int = 0,
) -> McResult:
    n = len(choices)
    if n <= 1:
        return McResult(pred_index=None, probs=None, meta={"reason": "not_enough_choices"})

    max_perms = max(1, int(max_perms))
    min_perms = max(1, min(int(min_perms), max_perms))

    letters = [chr(ord("A") + i) for i in range(n)]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    agg: list[float] = [0.0 for _ in range(n)]
    prev_dist: list[float] | None = None
    perms_used = 0

    for _ in range(max_perms):
        perm = torch.randperm(n, generator=g).tolist()
        perm_choices = [choices[i] for i in perm]
        messages = make_mc_messages(question, perm_choices, label_letters=letters)
        lps = batch_logprob_completions(model, messages, letters)

        # Map back to original option indices.
        for pos, orig_i in enumerate(perm):
            agg[orig_i] += float(lps[pos])

        perms_used += 1
        if perms_used < min_perms:
            continue

        dist = _softmax(agg)
        if prev_dist is not None and _js(prev_dist, dist) < js_eps:
            prev_dist = dist
            break
        prev_dist = dist

    if prev_dist is None:
        prev_dist = _softmax(agg)
    pred = max(range(n), key=lambda i: agg[i])
    return McResult(
        pred_index=pred,
        probs=prev_dist,
        meta={"perms_used": perms_used, "agg_logprobs": agg, "seed": seed},
    )


def latin_consensus(
    model: LoadedModel,
    question: str,
    choices: list[str],
    *,
    max_perms: int | None = None,
    min_perms: int = 2,
    js_eps: float = 0.005,
    seed: int = 0,
) -> McResult:
    """
    Position-balanced consensus with a cyclic Latin-square design.

    For n choices, we sample ONE random base permutation (seeded), then evaluate cyclic
    shifts of it. With n shifts, each option appears in every position exactly once.
    """

    n = len(choices)
    if n <= 1:
        return McResult(pred_index=None, probs=None, meta={"reason": "not_enough_choices"})

    letters = [chr(ord("A") + i) for i in range(n)]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    base = torch.randperm(n, generator=g).tolist()
    budget = int(max_perms) if max_perms is not None else n
    budget = max(1, min(budget, n))
    min_perms = max(1, min(int(min_perms), budget))

    agg: list[float] = [0.0 for _ in range(n)]
    prev_dist: list[float] | None = None
    perms_used = 0

    for shift in range(budget):
        perm = base[shift:] + base[:shift]
        perm_choices = [choices[i] for i in perm]
        messages = make_mc_messages(question, perm_choices, label_letters=letters)
        lps = batch_logprob_completions(model, messages, letters)

        for pos, orig_i in enumerate(perm):
            agg[orig_i] += float(lps[pos])

        perms_used += 1
        if perms_used < min_perms:
            continue

        dist = _softmax(agg)
        if prev_dist is not None and _js(prev_dist, dist) < js_eps:
            prev_dist = dist
            break
        prev_dist = dist

    if prev_dist is None:
        prev_dist = _softmax(agg)
    pred = max(range(n), key=lambda i: agg[i])
    return McResult(
        pred_index=pred,
        probs=prev_dist,
        meta={"perms_used": perms_used, "agg_logprobs": agg, "seed": seed, "design": "latin"},
    )
