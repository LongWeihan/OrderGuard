from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class LoadedModel:
    model_id: str
    tokenizer: Any
    model: Any
    device: str
    torch_dtype: str
    pad_to_multiple_of: int
    attn_implementation: str | None


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


def load_lm(
    model_id: str,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype | str | None = torch.float16,
    pad_to_multiple_of: int = 0,
    attn_implementation: str | None = None,
) -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    _ensure_pad_token(tokenizer)
    hf_dtype = "auto" if torch_dtype is None else torch_dtype
    extra = {}
    if attn_implementation:
        extra["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=hf_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            **extra,
        )
    except TypeError:
        # Some model classes / older Transformers may not accept attn_implementation.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=hf_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return LoadedModel(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
        torch_dtype=str(hf_dtype).replace("torch.", ""),
        pad_to_multiple_of=int(pad_to_multiple_of or 0),
        attn_implementation=attn_implementation,
    )


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer has no chat template; provide a chat/instruct model.")

    # Qwen3 defaults to thinking mode; for benchmarking MCQA/selection,
    # disabling it improves determinism and avoids long '<think>' outputs.
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return out


def chat_to_input_ids(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    out = _apply_chat_template(tokenizer, messages)
    if isinstance(out, torch.Tensor):
        input_ids = out
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    if isinstance(out, dict):
        input_ids = out["input_ids"]
        attention_mask = out.get("attention_mask", torch.ones_like(input_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    if hasattr(out, "input_ids"):
        input_ids = out.input_ids
        attention_mask = getattr(out, "attention_mask", torch.ones_like(input_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    raise TypeError(f"Unexpected apply_chat_template output type: {type(out)}")


def _round_up(x: int, m: int) -> int:
    if m <= 1:
        return x
    return ((x + m - 1) // m) * m


def _pad_to_batch(
    seqs: list[torch.Tensor],
    pad_token_id: int,
    *,
    pad_to_multiple_of: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(s.shape[1] for s in seqs)
    max_len = _round_up(max_len, int(pad_to_multiple_of or 0))
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : s.shape[1]] = s[0]
        attn[i, : s.shape[1]] = 1
    return input_ids, attn


@torch.inference_mode()
def batch_logprob_completions(
    loaded: LoadedModel,
    prompt_messages: list[dict[str, str]],
    completions: list[str],
) -> list[float]:
    """
    Return sum log-prob of each completion string (tokenized) conditioned on the prompt.
    """

    tok = loaded.tokenizer
    prompt_enc = chat_to_input_ids(tok, prompt_messages)
    prompt_ids = prompt_enc["input_ids"].to("cpu")
    prompt_len = prompt_ids.shape[1]

    seqs: list[torch.Tensor] = []
    comp_lens: list[int] = []
    for completion in completions:
        comp_ids = tok(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        comp_lens.append(comp_ids.shape[1])
        seqs.append(torch.cat([prompt_ids, comp_ids], dim=1))

    input_ids, attention_mask = _pad_to_batch(
        seqs,
        tok.pad_token_id,
        pad_to_multiple_of=loaded.pad_to_multiple_of,
    )
    input_ids = input_ids.to(loaded.device)
    attention_mask = attention_mask.to(loaded.device)

    logits = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]

    scores: list[float] = []
    for i, comp_len in enumerate(comp_lens):
        if comp_len <= 0:
            scores.append(float("-inf"))
            continue
        start = prompt_len - 1
        end = start + comp_len
        token_lp = logprobs[i, start:end, :].gather(dim=-1, index=targets[i, start:end].unsqueeze(-1))
        scores.append(float(token_lp.sum().item()))
    return scores
