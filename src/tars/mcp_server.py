"""MCP transport.

Same core, different consumer. The design difference that matters: an agent has
no scrollback and a finite context, so tools return identifiers and summaries,
and detail is a second, explicit call. Returning full evidence from a search
tool would blow the caller's context on the first query.

Run: python -m tars.mcp_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .deps import context, load_conversations

mcp = FastMCP("tars")


@mcp.tool()
def list_signals() -> dict:
    """List the quality signals this reviewer computes, with their scales.

    Call this first if you do not already know what the rubric measures.
    """
    rubric, *_ = context()
    return {
        "rubric_version": rubric.version,
        "signals": [
            {"name": s.name, "type": s.type, "scale": s.scale, "labels": s.labels,
             "criteria": s.criteria.strip()}
            for s in rubric.signals
        ],
    }


@mcp.tool()
def score_conversations(source: str = "synthetic", limit: int = 5) -> dict:
    """Score a batch of conversations. Returns a job id to poll, not results.

    source: 'abcd' or 'synthetic'. limit: how many conversations, max 100.
    """
    _, _, _, runner = context()
    convos = load_conversations(source, min(limit, 100))
    job = runner.submit(convos)
    return {"job_id": job.job_id, "submitted": len(convos),
            "next": "call check_job with this job_id"}


@mcp.tool()
def score_transcript(turns: list[dict]) -> dict:
    """Score a single transcript supplied inline. Returns the full result.

    turns: [{"speaker": "customer"|"agent"|"system", "text": "..."}] in order.
    Use this when the conversation is not already in a dataset.
    """
    from .api import InlineRequest, score_inline
    from .schema import Turn

    try:
        parsed = [Turn(idx=i, **t) for i, t in enumerate(turns)]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not parse turns: {exc}. "
                         "Each turn needs 'speaker' and 'text'."}
    result = score_inline(InlineRequest(turns=parsed))
    return result.model_dump(mode="json")


@mcp.tool()
def check_job(job_id: str) -> dict:
    """Check whether a scoring job has finished. Returns status and per-item counts."""
    _, _, _, runner = context()
    job = runner.get(job_id)
    if job is None:
        return {"error": f"no job with id {job_id}. Job ids look like job_abc123."}
    return {"job_id": job.job_id, "status": job.status, "counts": job.counts}


@mcp.tool()
def find_conversations(signal: str, max_score: float | None = None,
                       label: str | None = None, limit: int = 10) -> dict:
    """Find conversations that scored poorly on a signal.

    Returns ids and one-line summaries only. Call get_evidence for the quotes
    behind any single result.
    """
    rubric, store, _, _ = context()
    if rubric.signal(signal) is None:
        known = [s.name for s in rubric.signals]
        return {"error": f"unknown signal '{signal}'. Known signals: {known}"}

    rows = store.query(rubric_version=rubric.version, signal=signal,
                       max_score=max_score, label=label, limit=min(limit, 50))
    return {
        "count": len(rows),
        "results": [
            {"conversation_id": r.conversation_id,
             "score": r.signals[signal].score or r.signals[signal].label,
             "confidence": r.signals[signal].confidence}
            for r in rows
        ],
    }


@mcp.tool()
def get_evidence(conversation_id: str, signal: str) -> dict:
    """Get the transcript quotes that produced a given score.

    Use this to check a score before acting on it.
    """
    rubric, store, judge, _ = context()
    result = store.get(conversation_id, rubric.version, judge.model_version)
    if result is None:
        return {"error": f"{conversation_id} has not been scored under {rubric.version}"}
    sig = result.signals.get(signal)
    if sig is None:
        return {"error": f"signal '{signal}' not present on this result"}
    if sig.abstained:
        return {"abstained": True, "reason": sig.abstain_reason}
    return {
        "score": sig.score or sig.label,
        "confidence": sig.confidence,
        "rationale": sig.rationale,
        "evidence": [e.model_dump() for e in sig.evidence],
        "grounded": [g.model_dump() for g in result.grounded],
    }


if __name__ == "__main__":
    mcp.run()
