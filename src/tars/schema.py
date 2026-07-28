"""Canonical schema.

This module is the one-way door. Adapters write into these types, the judge
reads them, the store persists them, and every transport serialises them.
Adding optional fields is safe forever; renaming or restructuring breaks callers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class Speaker(str, Enum):
    CUSTOMER = "customer"
    AGENT = "agent"
    SYSTEM = "system"


class Turn(BaseModel):
    idx: int = Field(description="Zero-based position. Evidence points at this.")
    speaker: Speaker
    text: str
    ts: datetime | None = None


class Conversation(BaseModel):
    """The only shape the judge ever sees. Adapters exist to produce this."""

    conversation_id: str
    source: str = Field(description="Adapter that produced this, e.g. 'abcd'")
    turns: list[Turn]

    # Adapter-specific ground truth, kept out of the judge's view on purpose.
    # ABCD puts flow/subflow/executed actions here; Twitter would put brand.
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    def transcript(self) -> str:
        return "\n".join(f"[{t.idx}] {t.speaker.value}: {t.text}" for t in self.turns)


class Evidence(BaseModel):
    turn_idx: int
    quote: str = Field(max_length=300)
    reason: str


class SignalResult(BaseModel):
    """Mandatory evidence is a deliberate constraint.

    A signal may abstain, but it may not produce a bare number. Making evidence
    optional means a share of records can never be audited, and that is not
    recoverable after the fact.
    """

    name: str
    score: float | None = Field(default=None, description="None iff abstained")
    label: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    abstain_reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str = ""

    @property
    def is_scorable(self) -> bool:
        # Ordinal signals carry `score`, categorical signals carry `label`.
        # Checking only `score` silently reports every categorical signal as a
        # 100% abstention, which is exactly the kind of metric that looks
        # plausible on a dashboard and is completely wrong.
        return not self.abstained and (self.score is not None or self.label is not None)


class GroundedCheck(BaseModel):
    """Deterministic, non-LLM verification. Only some sources can supply this."""

    name: str
    passed: bool | None
    detail: str = ""


class ScoredConversation(BaseModel):
    """The result envelope. Once a caller reads this, the shape is frozen."""

    schema_version: str = SCHEMA_VERSION
    conversation_id: str
    source: str

    # Provenance. Without all three, a score is uninterpretable and cannot be
    # backfilled. This is the single most important field group in the system.
    rubric_version: str
    model_version: str

    signals: dict[str, SignalResult] = Field(default_factory=dict)
    grounded: list[GroundedCheck] = Field(default_factory=list)

    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    def idempotency_key(self) -> str:
        return f"{self.conversation_id}|{self.rubric_version}|{self.model_version}"


class Override(BaseModel):
    """Human disagreement. The calibration set and the eval set, later."""

    conversation_id: str
    rubric_version: str
    model_version: str
    signal: str
    # Ordinal signals correct a score; categorical signals correct a label.
    # Both pairs are optional so a reviewer only fills the one that applies.
    # Recording only scores meant a fully categorical rubric could never
    # register a disagreement, and agreement_rate read 1.0 forever.
    original_score: float | None = None
    corrected_score: float | None = None
    original_label: str | None = None
    corrected_label: str | None = None
    note: str = ""
    reviewer: str = "anonymous"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


JobStatus = Literal["queued", "running", "succeeded", "partial", "failed"]


class JobItem(BaseModel):
    conversation_id: str
    status: Literal["pending", "done", "error", "cached"] = "pending"
    error: str | None = None


class Job(BaseModel):
    job_id: str
    status: JobStatus = "queued"
    rubric_version: str
    model_version: str
    items: list[JobItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.items:
            out[i.status] = out.get(i.status, 0) + 1
        return out
