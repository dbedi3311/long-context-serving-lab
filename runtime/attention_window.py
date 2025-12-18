"""
Sliding attention window helpers.
"""
from __future__ import annotations

from typing import List, Sequence


class AttentionWindow:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def crop(self, tokens: Sequence[int]) -> List[int]:
        if len(tokens) <= self.window_size:
            return list(tokens)
        return list(tokens[-self.window_size :])

    def explain(self) -> str:
        return (
            f"A sliding window keeps the most recent {self.window_size} tokens, "
            "bounding compute while preserving local context."
        )
