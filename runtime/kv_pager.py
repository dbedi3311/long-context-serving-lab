"""
A minimal pager that places KV slots onto pages and can spill to CPU/SSD tiers.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

from runtime.eviction import LRUEvictionPolicy, evict_if_needed
from runtime.kv_types import KVSlot, Page, SessionKVState


class KVPager:
    def __init__(self, page_capacity: int = 4, max_pages: int = 32) -> None:
        self.page_capacity = page_capacity
        self.max_pages = max_pages
        self._page_ids = itertools.count(0)
        self._states: Dict[str, SessionKVState] = {}
        self._eviction = LRUEvictionPolicy()

    def ensure_session(self, session_id: str) -> SessionKVState:
        state = self._states.get(session_id)
        if state is None:
            state = SessionKVState(session_id=session_id)
            self._states[session_id] = state
        return state

    def append_tokens(self, session_id: str, token_ids: List[int], device: str = "gpu") -> Tuple[int, Page]:
        state = self.ensure_session(session_id)
        page_id = state.active_page
        page: Page
        if page_id is None or len(state.pages[page_id].slots) >= self.page_capacity:
            page_id = next(self._page_ids)
            page = Page(page_id=page_id, capacity=self.page_capacity, device=device)
            state.allocate_page(page)
        else:
            page = state.pages[page_id]
        slot = self._materialize_slot(token_ids, device)
        page.add_slot(slot)
        state.record_tokens(len(token_ids))
        self._eviction.touch(session_id, page.page_id)
        evict_if_needed(self._states, self.max_pages, self._eviction)
        return page.page_id, page

    def page_summary(self, session_id: str) -> List[dict]:
        state = self._states.get(session_id)
        if not state:
            return []
        summary = []
        for page in state.pages.values():
            summary.append(
                {
                    "page_id": page.page_id,
                    "device": page.device,
                    "slots": len(page.slots),
                    "bytes": page.residency_bytes(),
                }
            )
        return summary

    def _materialize_slot(self, token_ids: List[int], device: str) -> KVSlot:
        # A tiny random embedding to mimic KV tensors using pure Python lists.
        key = [[random.random() for _ in range(16)] for _ in token_ids]
        value = [[random.random() for _ in range(16)] for _ in token_ids]
        return KVSlot(key=key, value=value, token_ids=token_ids, device=device)

    def reset(self, session_id: str) -> None:
        if session_id in self._states:
            self._states.pop(session_id)
            self._eviction.reset_session(session_id)

    @property
    def states(self) -> Dict[str, SessionKVState]:
        return self._states
