"""
Simple eviction policies for KV paging. This module keeps the dependency footprint light
while providing realistic behavior for the lab's benchmarks.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from runtime.kv_types import Page, SessionKVState


@dataclass(order=True)
class _LRUEntry:
    last_used: float
    page_id: int = field(compare=False)
    session_id: str = field(compare=False)


class LRUEvictionPolicy:
    """Least-recently-used eviction across sessions."""

    def __init__(self) -> None:
        self._heap: list[_LRUEntry] = []
        self._index: Dict[Tuple[str, int], _LRUEntry] = {}

    def touch(self, session_id: str, page_id: int) -> None:
        now = time.time()
        key = (session_id, page_id)
        entry = _LRUEntry(last_used=now, page_id=page_id, session_id=session_id)
        self._index[key] = entry
        heapq.heappush(self._heap, entry)

    def evict(self) -> Optional[Tuple[str, int]]:
        while self._heap:
            entry = heapq.heappop(self._heap)
            key = (entry.session_id, entry.page_id)
            current = self._index.get(key)
            if current is entry:
                del self._index[key]
                return entry.session_id, entry.page_id
        return None

    def reset_session(self, session_id: str) -> None:
        """Remove all tracking for a terminated session."""
        for key in [k for k in self._index if k[0] == session_id]:
            del self._index[key]


def evict_if_needed(states: Dict[str, SessionKVState], max_pages: int, policy: LRUEvictionPolicy) -> None:
    """Evict oldest pages until the pool fits within the configured budget."""
    total_pages = sum(len(state.pages) for state in states.values())
    while total_pages > max_pages:
        victim = policy.evict()
        if victim is None:
            break
        session_id, page_id = victim
        session = states.get(session_id)
        if session and page_id in session.pages:
            session.pages.pop(page_id)
            total_pages -= 1
