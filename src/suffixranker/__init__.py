"""Public API for the `suffixranker` package.

We expose `ModelConfig` and `SuffixRanker` lazily so that importing
`suffixranker` (or `suffixranker.utils`) does not immediately pull in
heavy dependencies such as PyTorch.

This keeps lightweight utilities (e.g., `mapk`) usable in minimal
environments and in CI where GPU / CUDA / DLLs may not be available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["ModelConfig", "SuffixRanker"]

if TYPE_CHECKING:  # for static type checkers only
    # These imports are only for IDEs/mypy; they won't run at runtime.
    from .model import ModelConfig, SuffixRanker  # pragma: no cover


def __getattr__(name: str):
    """Lazily load heavy objects on first attribute access.

    When someone does:

        from suffixranker import ModelConfig, SuffixRanker

    we import them from `.model` at that moment. Importing
    `suffixranker.utils` or `suffixranker.data` will *not* trigger
    a PyTorch import anymore.
    """
    if name in __all__:
        from .model import ModelConfig, SuffixRanker

        mapping = {
            "ModelConfig": ModelConfig,
            "SuffixRanker": SuffixRanker,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
