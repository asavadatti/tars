"""HTTP contract.

One of three transports over the same core. REST, MCP, and the CLI all call
JobRunner and Store; none of them owns behaviour. That is what makes adding a
fourth transport cheap and changing the result envelope expensive.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .deps import context, find_conversation, load_conversations
from .grounding import run_checks
from .schema import Conversation, Job, Override, ScoredConversation, Speaker, Turn

TAGS = [
    {
        "name": "review",
        "description": "Score conversations and read the results. This is the surface a "
                       "reviewer, a dashboard, or an agent talks to.",
    },
    {
        "name": "operations",
        "description": "Batch orchestration and service health. Needed to run the thing at "
                       "volume, not to understand what it produces.",
    },
]

app = FastAPI(
    title="TARS — Conversation Quality Reviewer",
    version="0.1.0",
    description="Scores customer-support transcripts against a versioned rubric.",
    openapi_tags=TAGS,
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
    """What to score: either a transcript supplied inline, or a dataset id.

    Supplying `turns` scores them directly. Omitting `turns` and supplying
    `conversation_id` looks that id up in `source` and scores it. The two paths
    differ on caching, deliberately — see the endpoint description.
    """

    conversation_id: str | None = Field(
        None,
        description="With `turns`, the id to store the result under; omit for a content hash. "
                    "Without `turns`, the dataset id to look up and score.",
    )
    turns: list[InlineTurn] = Field(
        default_factory=list, description="Omit when scoring an existing conversation by id."
    )
    source: str = Field(
        "abcd", examples=["abcd", "synthetic"],
        description="Dataset to resolve `conversation_id` against. Ignored when `turns` is supplied.",
    )
    force: bool = Field(
        False, description="Lookup path only: re-score even if this triple is already stored."
    )


class BatchRequest(BaseModel):
    source: str = Field("synthetic", examples=["abcd", "synthetic"])
    limit: int = Field(10, ge=1, le=500)
    path: str | None = None
    force: bool = Field(False, description="Bypass the idempotency cache.")


class SignalSummary(BaseModel):
    """One signal with its evidence stripped."""

    value: float | str | None = Field(
        description="The label for a categorical signal, the score for an ordinal one, null if abstained."
    )
    confidence: float
    abstained: bool


class ResultSummary(BaseModel):
    """Scores without evidence, for callers that will not read the quotes.

    Two fields are carried even though they are not scores. The provenance pair,
    because a score with no rubric and model attached cannot be interpreted or
    compared later — that is the whole reason the store keys on it. And `error`,
    because a failed run is persisted with an empty `signals` map: without the
    error field a projection would answer 200 with `{}` and give the caller no
    way to tell "scored nothing" from "the API key was rejected".
    """

    conversation_id: str
    rubric_version: str
    model_version: str
    signals: dict[str, SignalSummary]
    error: str | None = None

    @classmethod
    def of(cls, result: ScoredConversation) -> "ResultSummary":
        return cls(
            conversation_id=result.conversation_id,
            rubric_version=result.rubric_version,
            model_version=result.model_version,
            signals={
                name: SignalSummary(
                    value=None if s.abstained else (s.label if s.label is not None else s.score),
                    confidence=s.confidence,
                    abstained=s.abstained,
                )
                for name, s in result.signals.items()
            },
            error=result.error,
        )


SCORE_EXAMPLES = {
    "inline": {
        "summary": "Score a transcript inline",
        "description": "Always calls the model. Use this to prove the system is live.",
        "value": {
            "turns": [
                {"speaker": "customer", "text": "Third time I have contacted you about this charge. I am done."},
                {"speaker": "agent", "text": "Please provide your order number."},
                {"speaker": "customer", "text": "8812."},
                {"speaker": "agent", "text": "I have refunded it. Anything else?"},
            ]
        },
    },
    "by_id": {
        "summary": "Score a dataset conversation by id",
        "description": "Returns the stored result when one exists under this rubric and model.",
        "value": {"conversation_id": "abcd-9489", "source": "abcd"},
    },
    "by_id_force": {
        "summary": "Re-score by id, bypassing the cache",
        "description": "Calls the model even when a result is already stored.",
        "value": {"conversation_id": "abcd-9489", "source": "abcd", "force": True},
    },
}


@app.post("/v1/score", response_model=ScoredConversation | ResultSummary, tags=["review"],
          summary="Score one conversation synchronously, inline or by dataset id")
def score_inline(
    req: InlineRequest = Body(openapi_examples=SCORE_EXAMPLES),
    include: Literal["all", "signals"] = "all",
):
    """Score a conversation by passing in the conversation text or the canonical conversation id.

    You can also rescore a previous conversation (if the model or rubric has changed for example).

    You can supply the `conversation_id` to score the exact conversation or add a new conversation inline to score.
    The new conversation must be in the correct JSON format.
    """
    rubric, store, judge, _ = context()
    grounded_by = [s.grounded_by for s in rubric.signals if s.grounded_by]

    if req.turns:
        turns = [Turn(idx=i, speaker=t.speaker, text=t.text) for i, t in enumerate(req.turns)]
        convo_id = req.conversation_id or "inline-" + hashlib.sha256(
            "".join(f"{t.speaker.value}:{t.text}" for t in turns).encode()
        ).hexdigest()[:12]
        convo = Conversation(conversation_id=convo_id, source="inline", turns=turns)
    elif req.conversation_id:
        convo = find_conversation(req.source, req.conversation_id)
        if convo is None:
            raise HTTPException(
                status_code=404,
                detail=f"no conversation {req.conversation_id!r} in source {req.source!r}",
            )
        if not req.force:
            cached = store.get(convo.conversation_id, rubric.version, judge.model_version)
            if cached is not None:
                return ResultSummary.of(cached) if include == "signals" else cached
    else:
        raise HTTPException(
            status_code=400, detail="supply either turns or conversation_id"
        )

    result = judge.score(convo, rubric)
    # Same grounding the batch path runs. Without this, scoring a corpus
    # conversation here would overwrite its stored record with one missing the
    # deterministic checks — the same triple, quietly worth less.
    result.grounded = run_checks(convo, grounded_by)
    store.put(result)
    return ResultSummary.of(result) if include == "signals" else result


@app.get("/v1/rubric", tags=["review"], summary="Active rubric and its version")
def get_rubric():
    rubric, *_ = context()
    return {"version": rubric.version, "signals": [s.model_dump() for s in rubric.signals]}


@app.post("/v1/score:batch", response_model=Job, status_code=202, tags=["operations"],
          summary="Submit a batch; returns immediately with a job id")
def score_batch(req: BatchRequest) -> Job:
    _, _, _, runner = context()
    try:
        convos = load_conversations(req.source, req.limit, req.path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runner.submit(convos, force=req.force)


@app.get("/v1/jobs/{job_id}", response_model=Job, tags=["operations"],
         summary="Poll job status")
def get_job(job_id: str) -> Job:
    _, _, _, runner = context()
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@app.get("/v1/results/{conversation_id}", response_model=ScoredConversation | ResultSummary,
         tags=["review"],
         summary="One result; full envelope by default, scores only with include=signals")
def get_result(conversation_id: str, rubric_version: str | None = None,
               model_version: str | None = None,
               include: Literal["all", "signals"] = "all"):
    """Fetch one scored conversation.

    `include=all` returns the whole envelope: every signal with its evidence
    quotes, the deterministic grounded checks, and the token and latency counts.

    `include=signals` drops the evidence and returns roughly a fifteenth of the
    bytes. It is for callers that aggregate — a dashboard plotting scores across
    a corpus reads no quotes. Provenance and `error` are carried regardless, so
    a projected result is never less interpretable than the full one.

    404 means no record under this exact `(conversation_id, rubric_version,
    model_version)` triple. That covers four different situations — never
    submitted, still running, or scored under a different rubric or model — so
    check `/v1/jobs/{id}` before concluding a conversation was never seen.
    """
    rubric, store, judge, _ = context()
    result = store.get(conversation_id, rubric_version or rubric.version,
                       model_version or judge.model_version)
    if result is None:
        raise HTTPException(status_code=404, detail="not scored under this rubric and model")
    return ResultSummary.of(result) if include == "signals" else result


@app.get("/v1/results", tags=["review"], summary="Filter by signal; thresholds applied at read time")
def search(signal: str | None = None, max_score: float | None = None,
           label: str | None = None, limit: int = 20):
    rubric, store, _, _ = context()
    rows = store.query(rubric_version=rubric.version, signal=signal,
                       max_score=max_score, label=label, limit=limit)
    return {"count": len(rows), "results": rows}


@app.post("/v1/overrides", status_code=201, tags=["review"],
          summary="Record a human correction; this is the calibration set")
def add_override(override: Override):
    _, store, _, _ = context()
    store.add_override(override)
    return {"ok": True}


@app.get("/v1/metrics", tags=["operations"],
         summary="Whether the thing is working, not what it output")
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
