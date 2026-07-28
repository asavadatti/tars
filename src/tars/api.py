"""HTTP contract.

One of three transports over the same core. REST, MCP, and the CLI all call
JobRunner and Store; none of them owns behaviour. That is what makes adding a
fourth transport cheap and changing the result envelope expensive.
"""

from __future__ import annotations

import hashlib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .deps import context, load_conversations
from .schema import Conversation, Job, Override, ScoredConversation, Speaker, Turn

app = FastAPI(
    title="TARS — Conversation Quality Reviewer",
    version="0.1.0",
    description="Scores customer-support transcripts against a versioned rubric.",
)


class InlineTurn(BaseModel):
    """A turn as a caller supplies it.

    No `idx`: position in the list is the order, and the endpoint assigns
    indices. Requiring callers to number their own turns invites off-by-one
    evidence pointers, which are worse than useless because they look valid.
    """

    speaker: Speaker
    text: str


class InlineRequest(BaseModel):
    """A transcript supplied directly, not read from a dataset file."""

    conversation_id: str | None = Field(
        None, description="Omit to get a content-hashed id. Supply one to make re-runs cacheable."
    )
    turns: list[InlineTurn]


class BatchRequest(BaseModel):
    source: str = Field("synthetic", examples=["abcd", "synthetic"])
    limit: int = Field(10, ge=1, le=500)
    path: str | None = None
    force: bool = Field(False, description="Bypass the idempotency cache.")


@app.post("/v1/score", response_model=ScoredConversation,
          summary="Score one transcript synchronously")
def score_inline(req: InlineRequest) -> ScoredConversation:
    """Synchronous on purpose, and the only endpoint that is.

    One conversation is a single model call, so the async ceremony buys nothing
    and costs a round trip. Batches stay async because they are the case that
    cannot be made synchronous later without breaking callers.
    """
    rubric, store, judge, _ = context()
    if not req.turns:
        raise HTTPException(status_code=400, detail="turns must not be empty")

    turns = [Turn(idx=i, speaker=t.speaker, text=t.text) for i, t in enumerate(req.turns)]
    convo_id = req.conversation_id or "inline-" + hashlib.sha256(
        "".join(f"{t.speaker.value}:{t.text}" for t in turns).encode()
    ).hexdigest()[:12]

    convo = Conversation(conversation_id=convo_id, source="inline", turns=turns)
    result = judge.score(convo, rubric)
    store.put(result)
    return result


@app.get("/v1/rubric", summary="Active rubric and its version")
def get_rubric():
    rubric, *_ = context()
    return {"version": rubric.version, "signals": [s.model_dump() for s in rubric.signals]}


@app.post("/v1/score:batch", response_model=Job, status_code=202,
          summary="Submit a batch; returns immediately with a job id")
def score_batch(req: BatchRequest) -> Job:
    _, _, _, runner = context()
    try:
        convos = load_conversations(req.source, req.limit, req.path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.submit(convos, force=req.force)


@app.get("/v1/jobs/{job_id}", response_model=Job, summary="Poll job status")
def get_job(job_id: str) -> Job:
    _, _, _, runner = context()
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@app.get("/v1/results/{conversation_id}", response_model=ScoredConversation,
         summary="Full result including evidence")
def get_result(conversation_id: str, rubric_version: str | None = None,
               model_version: str | None = None) -> ScoredConversation:
    rubric, store, judge, _ = context()
    result = store.get(conversation_id, rubric_version or rubric.version,
                       model_version or judge.model_version)
    if result is None:
        raise HTTPException(status_code=404, detail="not scored under this rubric and model")
    return result


@app.get("/v1/results", summary="Filter by signal; thresholds applied at read time")
def search(signal: str | None = None, max_score: float | None = None,
           label: str | None = None, limit: int = 20):
    rubric, store, _, _ = context()
    rows = store.query(rubric_version=rubric.version, signal=signal,
                       max_score=max_score, label=label, limit=limit)
    return {"count": len(rows), "results": rows}


@app.post("/v1/overrides", status_code=201,
          summary="Record a human correction; this is the calibration set")
def add_override(override: Override):
    _, store, _, _ = context()
    store.add_override(override)
    return {"ok": True}


@app.get("/v1/metrics", summary="Whether the thing is working, not what it output")
def metrics():
    rubric, store, _, _ = context()
    rows = store.query(rubric_version=rubric.version, limit=10_000)
    per_signal: dict[str, dict[str, float]] = {}
    for spec in rubric.signals:
        vals = [r.signals[spec.name] for r in rows if spec.name in r.signals]
        scored = [v for v in vals if v.is_scorable]
        per_signal[spec.name] = {
            "n": len(vals),
            "abstention_rate": round(1 - len(scored) / len(vals), 3) if vals else 0.0,
            "mean_confidence": round(sum(v.confidence for v in vals) / len(vals), 3) if vals else 0.0,
            "mean_evidence_count": round(sum(len(v.evidence) for v in vals) / len(vals), 2) if vals else 0.0,
        }
    return {
        "rubric_version": rubric.version,
        "scored": len(rows),
        "error_rate": round(sum(1 for r in rows if r.error) / len(rows), 3) if rows else 0.0,
        "signals": per_signal,
        "agreement": store.agreement_rate(rubric.version),
    }
