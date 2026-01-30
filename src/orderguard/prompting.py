from __future__ import annotations


def make_mc_messages(question: str, choices: list[str], *, label_letters: list[str]) -> list[dict[str, str]]:
    options = "\n".join(f"{lab}. {txt}" for lab, txt in zip(label_letters, choices, strict=True))
    user = (
        "Answer the multiple-choice question. Reply with ONLY the single letter.\n\n"
        f"Question:\n{question}\n\nChoices:\n{options}\n\nAnswer (letter only):"
    )
    return [
        {"role": "system", "content": "You answer multiple-choice questions."},
        {"role": "user", "content": user},
    ]

