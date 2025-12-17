"""
FastAPI application simulating long-context KV paging across sessions.
"""
from __future__ import annotations

from typing import Dict, List

try:  # Optional dependency; provide a lightweight fallback.
    from fastapi import FastAPI, HTTPException
except Exception:  # pragma: no cover - offline fallback
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:  # type: ignore[misc]
        def __init__(self, title: str = "") -> None:
            self.title = title
            self.routes = {}

        def post(self, path: str):
            def decorator(fn):
                self.routes[("POST", path)] = fn
                return fn

            return decorator

        def get(self, path: str):
            def decorator(fn):
                self.routes[("GET", path)] = fn
                return fn

            return decorator

try:  # Optional dependency; provide a thin BaseModel stub when missing.
    from pydantic import BaseModel
except Exception:  # pragma: no cover - offline fallback
    class BaseModel:  # type: ignore[misc]
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

from runtime.kv_pager import KVPager
from server.session import SessionStore


app = FastAPI(title="Long Context Serving Lab")
sessions = SessionStore()


class StartRequest(BaseModel):
    session_id: str


class StepRequest(BaseModel):
    session_id: str
    prompt_tokens: List[int]


@app.post("/start")
def start(req: StartRequest) -> Dict[str, str]:
    sessions.start(req.session_id)
    return {"status": "started", "session_id": req.session_id}


@app.post("/step")
def step(req: StepRequest) -> Dict[str, object]:
    try:
        session = sessions.get(req.session_id)
    except KeyError as exc:  # pragma: no cover - API surface
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.append(req.prompt_tokens)
    proposal = session.speculative.generate(len(session.tokens))
    page_summary = session.pager.page_summary(req.session_id)
    return {
        "accepted_tokens": proposal.tokens,
        "draft_batch": session.speculative.draft_batch,
        "pages": page_summary,
        "window": session.context_view(window=128),
    }


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# Convenience for `python -m server.app` during local development.
if __name__ == "__main__":  # pragma: no cover
    try:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception:
        print("uvicorn not installed; run FastAPI under a full environment to serve HTTP.")
