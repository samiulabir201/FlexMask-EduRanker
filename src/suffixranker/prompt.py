from __future__ import annotations

USER_START = "<|im_start|>user"
ASSISTANT_START = "<|im_start|>assistant"
IM_END = "<|im_end|>"


def build_prefix(bundle: dict[str, str]) -> str:
    """Render the shared question–answer context shown to the model.

    Args:
        bundle: Dictionary containing fields like:
            - "QuestionText"
            - "MC_Choices"
            - "Answer"
            - "MisconceptionCandidates"
            - "MC_Answer"
            - "StudentExplanation"

    Returns:
        A formatted prefix string using the chat-style template consumed by the LLM.
    """
    question = bundle.get("QuestionText", "")
    choices = bundle.get("MC_Choices", "")
    answer = bundle.get("Answer", "")
    misconceptions = bundle.get("MisconceptionCandidates", "")
    student_answer = bundle.get("MC_Answer", "")
    explanation = bundle.get("StudentExplanation", "")

    return (
        f"{USER_START}\n"
        f"**Question:** {question}\n"
        f"**Choices:** {choices}\n"
        f"**Correct Answer:** {answer}\n"
        f"**Common Misconceptions:** {misconceptions}\n"
        f"**Student Answer:** {student_answer}\n"
        f"**Student Explanation:** {explanation}\n"
        f"{IM_END}\n"
        f"{ASSISTANT_START}\n"
    )


def render_sample(prefix: str, suffixes: list[str]) -> list[str]:
    """Combine a shared prefix with each candidate suffix.

    Args:
        prefix: Shared prompt prefix (question + student response context).
        suffixes: Candidate `Category:Misconception` suffix strings.

    Returns:
        List of full prompt strings, one per suffix.
    """
    return [prefix + suffix for suffix in suffixes]
