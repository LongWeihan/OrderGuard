from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset


@dataclass(frozen=True)
class McTask:
    name: str
    hf_path: str
    hf_config: str | None
    split: str
    default_limit: int | None
    # Returns (question, choices_texts, gold_index)
    to_mc: Callable[[dict[str, Any]], tuple[str, list[str], int]]


def _arc_challenge_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["question"].strip()
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    ans = ex["answerKey"].strip().upper()
    gold = labels.index(ans)
    return q, list(texts), gold


def _arc_easy_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["question"].strip()
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    ans = ex["answerKey"].strip().upper()
    gold = labels.index(ans)
    return q, list(texts), gold


def _openbookqa_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["question_stem"].strip()
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    ans = ex["answerKey"].strip().upper()
    gold = labels.index(ans)
    return q, list(texts), gold


def _commonsenseqa_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["question"].strip()
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    ans = ex["answerKey"].strip().upper()
    gold = labels.index(ans)
    return q, list(texts), gold


def _hellaswag_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["ctx"].strip()
    choices = [c.strip() for c in ex["endings"]]
    gold = int(ex["label"])
    return q, choices, gold


def _winogrande_xs_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    # Sentence contains "_" placeholder.
    sent = ex["sentence"].strip()
    q = sent.replace("_", "____")
    choices = [ex["option1"].strip(), ex["option2"].strip()]
    gold = int(ex["answer"]) - 1
    return q, choices, gold


def _boolq_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    passage = ex["passage"].strip()
    q = ex["question"].strip()
    question = f"Passage:\n{passage}\n\nQuestion:\n{q}"
    choices = ["Yes", "No"]
    gold = 0 if bool(ex["answer"]) else 1
    return question, choices, gold


def _mmlu_all_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["question"].strip()
    choices = [c.strip() for c in ex["choices"]]
    gold = int(ex["answer"])
    return q, choices, gold


def _truthfulqa_mc1_to_mc(ex: dict[str, Any]) -> tuple[str, list[str], int]:
    q = ex["question"].strip()
    choices = [c.strip() for c in ex["mc1_targets"]["choices"]]
    labels = list(ex["mc1_targets"]["labels"])
    if not labels:
        raise ValueError("truthfulqa_mc1: empty labels")
    gold = int(max(range(len(labels)), key=lambda i: labels[i]))
    return q, choices, gold


TASKS: dict[str, McTask] = {
    "arc_challenge": McTask(
        name="arc_challenge",
        hf_path="ai2_arc",
        hf_config="ARC-Challenge",
        split="test",
        default_limit=None,
        to_mc=_arc_challenge_to_mc,
    ),
    "arc_easy": McTask(
        name="arc_easy",
        hf_path="ai2_arc",
        hf_config="ARC-Easy",
        split="test",
        default_limit=None,
        to_mc=_arc_easy_to_mc,
    ),
    "openbookqa": McTask(
        name="openbookqa",
        hf_path="openbookqa",
        hf_config="main",
        split="test",
        default_limit=None,
        to_mc=_openbookqa_to_mc,
    ),
    "commonsenseqa": McTask(
        name="commonsenseqa",
        hf_path="commonsense_qa",
        hf_config=None,
        split="validation",
        default_limit=None,
        to_mc=_commonsenseqa_to_mc,
    ),
    "hellaswag": McTask(
        name="hellaswag",
        hf_path="hellaswag",
        hf_config=None,
        split="validation",
        default_limit=1000,
        to_mc=_hellaswag_to_mc,
    ),
    "winogrande_xs": McTask(
        name="winogrande_xs",
        hf_path="winogrande",
        hf_config="winogrande_xs",
        split="validation",
        default_limit=None,
        to_mc=_winogrande_xs_to_mc,
    ),
    "boolq": McTask(
        name="boolq",
        hf_path="boolq",
        hf_config=None,
        split="validation",
        default_limit=None,
        to_mc=_boolq_to_mc,
    ),
    "mmlu_all": McTask(
        name="mmlu_all",
        hf_path="cais/mmlu",
        hf_config="all",
        split="validation",
        default_limit=None,
        to_mc=_mmlu_all_to_mc,
    ),
    "truthfulqa_mc1": McTask(
        name="truthfulqa_mc1",
        hf_path="truthful_qa",
        hf_config="multiple_choice",
        split="validation",
        default_limit=None,
        to_mc=_truthfulqa_mc1_to_mc,
    ),
}


def load_task(task_name: str):
    spec = TASKS[task_name]
    return load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
