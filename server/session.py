"""
Session state for the demo FastAPI server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from runtime.attention_window import AttentionWindow
from runtime.kv_pager import KVPager
from runtime.speculative import SpeculativeDecoder


@dataclass
class SessionState:
    session_id: str
    pager: KVPager
    attention: AttentionWindow
    speculative: SpeculativeDecoder
    tokens: List[int] = field(default_factory=list)

    def append(self, token_ids: List[int]) -> None:
        self.tokens.extend(token_ids)
        self.pager.append_tokens(self.session_id, token_ids)

    def context_view(self, window: int) -> List[int]:
        return self.attention.crop(self.tokens)[-window:]


class SessionStore:
    def __init__(self, window_size: int = 1024) -> None:
        self.window_size = window_size
        self.sessions: Dict[str, SessionState] = {}

    def start(self, session_id: str) -> SessionState:
        state = SessionState(
            session_id=session_id,
            pager=KVPager(),
            attention=AttentionWindow(self.window_size),
            speculative=SpeculativeDecoder(),
        )
        self.sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session {session_id}")
        return self.sessions[session_id]
