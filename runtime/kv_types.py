"""
Core KV cache data structures for the long-context serving lab.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class KVSlot:
    """Represents a contiguous slice of the KV cache (a few tokens)."""

    key: List[List[float]]
    value: List[List[float]]
    token_ids: List[int]
    device: str = "cpu"

    def bytes(self) -> int:
        """Approximate memory footprint for reporting/eviction decisions."""
        # Assume float32 entries
        return int((len(self.key) * len(self.key[0]) + len(self.value) * len(self.value[0])) * 4)


@dataclass
class Page:
    """A page stores multiple KV slots and tracks residency."""

    page_id: int
    capacity: int
    device: str
    slots: List[KVSlot] = field(default_factory=list)
    pinned: bool = False

    def add_slot(self, slot: KVSlot) -> None:
        if len(self.slots) >= self.capacity:
            raise ValueError("Page is full")
        self.slots.append(slot)

    def residency_bytes(self) -> int:
        return sum(slot.bytes() for slot in self.slots)


@dataclass
class SessionKVState:
    """Tracking paged KV allocations per session."""

    session_id: str
    pages: Dict[int, Page] = field(default_factory=dict)
    active_page: Optional[int] = None
    total_tokens: int = 0

    def allocate_page(self, page: Page) -> None:
        self.pages[page.page_id] = page
        self.active_page = page.page_id

    def record_tokens(self, count: int) -> None:
        self.total_tokens += count

    def window_tokens(self, window: int) -> List[int]:
        """Return the most recent token ids within a sliding window."""
        tokens: List[int] = []
        for page in self.pages.values():
            for slot in page.slots:
                tokens.extend(slot.token_ids)
        return tokens[-window:]
