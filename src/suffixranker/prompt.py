# -*- coding: utf-8 -*-
"""Prompt formatting utilities: prefix and suffix rendering."""

from __future__ import annotations
from typing import Dict, List

USER_START = "<|im_start|>user"
ASSISTANT_START = "<|im_start|>assistant"
IM_END = "<|im_end|>"

def build_prefix(bundle: Dict[str, str]) -> str:
    """Render the prefix block shown to the model.

    Args:
        bundle: A mapping with the expected fields: question, choices, answer,
            misconceptions, student_answer, student_explanation.

    Returns:
        The prefix text.
    """
    return (
        f"{USER_START}\\n"
        f"**Question:** {bundle['question']}\\n"
        f"**Choices:** {bundle['choices']}\\n"
        f"**Correct Answer:** {bundle['answer']}\\n"
        f"**Common Misconceptions:** {bundle['misconceptions']}\\n"
        f"**Student Answer:** {bundle['student_answer']}\\n"
        f"**Student Explanation:** {bundle['student_explanation']}\\n"
        f"{IM_END}\\n"
        f"{ASSISTANT_START}\\n"
    )

def render_sample(prefix: str, suffixes: List[str]) -> List[str]:
    """Combine prefix with each suffix candidate for per-candidate scoring.

    Args:
        prefix: The rendered prefix string.
        suffixes: List of candidate suffix strings.

    Returns:
        A list of full strings: [prefix + suffix_i for i in candidates].
    """
    return [prefix + s for s in suffixes]
