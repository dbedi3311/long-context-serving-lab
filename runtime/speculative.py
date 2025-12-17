"""
Speculative decoding toy implementation to demonstrate latency improvements.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DraftProposal:
    tokens: List[int]
    accepted: int
    rejected: int


class SpeculativeDecoder:
    def __init__(self, draft_batch: int = 4, accept_prob: float = 0.7) -> None:
        self.draft_batch = draft_batch
        self.accept_prob = accept_prob

    def generate(self, prompt_len: int) -> DraftProposal:
        proposal = [random.randint(1, 32000) for _ in range(self.draft_batch)]
        accepted = 0
        for _ in proposal:
            if random.random() < self.accept_prob:
                accepted += 1
        rejected = self.draft_batch - accepted
        return DraftProposal(tokens=proposal[:accepted], accepted=accepted, rejected=rejected)

    def latency_breakdown(self, prompt_len: int) -> Tuple[float, float]:
        """Return (draft_time_ms, verify_time_ms)."""
        draft_time = max(1.0, prompt_len * 0.01)
        verify_time = max(1.0, draft_time * 0.5)
        return draft_time, verify_time
