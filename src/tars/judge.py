"""Judge engine.

Swappable by design. The model id, the prompt wording, and the provider all sit
behind this interface, which is why changing any of them is a two-way door.

Structured output is obtained via a forced tool call rather than by asking for
JSON in prose. Prose JSON fails at a rate that is low enough to look fine in a
demo and high enough to be a real operational cost at volume.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Protocol

from .rubric import Rubric, SignalSpec
from .schema import Conversation, Evidence, ScoredConversation, SignalResult

SYSTEM_PROMPT = """You review customer-support conversation transcripts against a fixed rubric.

Rules you must follow:
- Score only what the transcript shows. Do not infer intent that is not present in the words.
- Every score requires evidence: at least one turn index and a short verbatim quote.
- If the transcript does not contain enough information to judge a signal, abstain.
  Abstaining is always correct when the evidence is absent. A guessed score is worse
  than no score, because a guessed score is indistinguishable from a real one downstream.
- Turns marked `system` are automated action records, not utterances. Use them as
  context for what happened, never as evidence of what the agent said.
- Judge the agent's conduct, not the customer's."""


def _signal_schema(spec: SignalSpec, max_evidence: int) -> dict[str, Any]:
    score: dict[str, Any]
    if spec.type == "ordinal":
        lo, hi = (spec.scale or [1, 5])[0], (spec.scale or [1, 5])[-1]
        score = {"type": "number", "minimum": lo, "maximum": hi,
                 "description": f"null if abstaining, otherwise {lo}..{hi}"}
    else:
        score = {"type": "string", "enum": list(spec.labels or []),
                 "description": "null if abstaining"}

    return {
        "type": "object",
        "properties": {
            "score": score,
            "abstained": {"type": "boolean"},
            "abstain_reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string", "description": "One or two sentences."},
            "evidence": {
                "type": "array",
                "maxItems": max_evidence,
                "items": {
                    "type": "object",
                    "properties": {
                        "turn_idx": {"type": "integer"},
                        "quote": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["turn_idx", "quote", "reason"],
                },
            },
        },
        "required": ["abstained", "confidence", "rationale", "evidence"],
    }


def build_tool(rubric: Rubric) -> dict[str, Any]:
    return {
        "name": "record_review",
        "description": "Record the quality review for this conversation.",
        "input_schema": {
            "type": "object",
            "properties": {
                s.name: _signal_schema(s, rubric.judge.max_evidence_per_signal)
                for s in rubric.signals
            },
            "required": [s.name for s in rubric.signals],
        },
    }


def build_prompt(convo: Conversation, rubric: Rubric) -> str:
    lines = ["Rubric:", ""]
    for s in rubric.signals:
        lines.append(f"## {s.name}")
        if s.type == "ordinal":
            lines.append(f"Scale: {s.scale[0]} to {s.scale[-1]}")
        else:
            lines.append(f"Allowed labels: {', '.join(s.labels or [])}")
        lines.append(s.criteria.strip())
        if s.abstain_when:
            lines.append(f"Abstain when: {s.abstain_when.strip()}")
        lines.append("")
    lines += ["Transcript:", "", convo.transcript()]
    return "\n".join(lines)


class Judge(Protocol):
    @property
    def model_version(self) -> str: ...
    def score(self, convo: Conversation, rubric: Rubric) -> ScoredConversation: ...


def _to_signal(name: str, raw: dict[str, Any]) -> SignalResult:
    raw_score = raw.get("score")
    abstained = bool(raw.get("abstained")) or raw_score is None
    numeric = raw_score if isinstance(raw_score, (int, float)) else None
    label = raw_score if isinstance(raw_score, str) else None
    return SignalResult(
        name=name,
        score=None if abstained else numeric,
        label=None if abstained else label,
        confidence=float(raw.get("confidence", 0.0)),
        abstained=abstained,
        abstain_reason=raw.get("abstain_reason") if abstained else None,
        rationale=raw.get("rationale", ""),
        evidence=[Evidence(**e) for e in raw.get("evidence", []) if isinstance(e, dict)],
    )


class AnthropicJudge:
    def __init__(self, rubric: Rubric, api_key: str | None = None, max_retries: int = 3):
        from anthropic import Anthropic  # imported lazily so the stub path needs no SDK

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = rubric.judge.model
        self._max_retries = max_retries

    @property
    def model_version(self) -> str:
        return self._model

    def score(self, convo: Conversation, rubric: Rubric) -> ScoredConversation:
        tool = build_tool(rubric)
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=rubric.judge.max_tokens,
                    temperature=rubric.judge.temperature,
                    system=SYSTEM_PROMPT,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": build_prompt(convo, rubric)}],
                )
                block = next(b for b in resp.content if b.type == "tool_use")
                payload: dict[str, Any] = block.input

                return ScoredConversation(
                    conversation_id=convo.conversation_id,
                    source=convo.source,
                    rubric_version=rubric.version,
                    model_version=self._model,
                    signals={
                        s.name: _to_signal(s.name, payload.get(s.name, {}))
                        for s in rubric.signals
                    },
                    latency_ms=int((time.monotonic() - started) * 1000),
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last_error = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt + random.random())

        return ScoredConversation(
            conversation_id=convo.conversation_id,
            source=convo.source,
            rubric_version=rubric.version,
            model_version=self._model,
            latency_ms=int((time.monotonic() - started) * 1000),
            error=f"{type(last_error).__name__}: {last_error}",
        )


class StubJudge:
    """Deterministic fake. Lets the whole pipeline run with no key and no network.

    Not a mock for tests only — it is what makes the demo survive a bad room.
    """

    def __init__(self, rubric: Rubric):
        self._rubric = rubric

    @property
    def model_version(self) -> str:
        return "stub-0"

    def score(self, convo: Conversation, rubric: Rubric) -> ScoredConversation:
        rng = random.Random(convo.conversation_id)
        signals = {}
        for s in rubric.signals:
            abstain = rng.random() < 0.15
            if s.type == "ordinal":
                lo, hi = (s.scale or [1, 5])[0], (s.scale or [1, 5])[-1]
                value = float(rng.randint(int(lo), int(hi)))
                signals[s.name] = SignalResult(
                    name=s.name, score=None if abstain else value,
                    confidence=round(rng.uniform(0.5, 0.95), 2), abstained=abstain,
                    abstain_reason="stub abstention" if abstain else None,
                    rationale="stub judge output",
                    evidence=[] if abstain else [Evidence(turn_idx=0, quote=convo.turns[0].text[:60], reason="stub")],
                )
            else:
                label = rng.choice(s.labels or ["unknown"])
                signals[s.name] = SignalResult(
                    name=s.name, label=None if abstain else label,
                    confidence=round(rng.uniform(0.5, 0.95), 2), abstained=abstain,
                    abstain_reason="stub abstention" if abstain else None,
                    rationale="stub judge output",
                    evidence=[] if abstain else [Evidence(turn_idx=0, quote=convo.turns[0].text[:60], reason="stub")],
                )
        return ScoredConversation(
            conversation_id=convo.conversation_id, source=convo.source,
            rubric_version=rubric.version, model_version=self.model_version,
            signals=signals, latency_ms=1,
        )


def get_judge(rubric: Rubric, stub: bool = False) -> Judge:
    if stub or not os.environ.get("ANTHROPIC_API_KEY"):
        return StubJudge(rubric)
    return AnthropicJudge(rubric)
