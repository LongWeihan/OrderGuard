from __future__ import annotations

import re

_MC_LETTER_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def normalize_mc_letter(text: str) -> str | None:
    match = _MC_LETTER_RE.search(text.strip())
    if not match:
        return None
    return match.group(1).upper()

