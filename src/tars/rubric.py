from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SignalSpec(BaseModel):
    name: str
    type: Literal["ordinal", "categorical"]
    criteria: str
    scale: list[float] | None = None
    labels: list[str] | None = None
    abstain_when: str | None = None
    grounded_by: str | None = None


class JudgeSpec(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    temperature: float = 0.0
    max_evidence_per_signal: int = 3


class Rubric(BaseModel):
    name: str
    revision: int
    description: str = ""
    signals: list[SignalSpec]
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    content_hash: str = ""

    @property
    def version(self) -> str:
        """Name, revision, and a content hash.

        The hash is the part that matters: it catches the case where someone
        edits criteria text without bumping the revision, which would otherwise
        silently poison every trendline drawn across the change.
        """
        return f"{self.name}.v{self.revision}+{self.content_hash[:8]}"

    def signal(self, name: str) -> SignalSpec | None:
        return next((s for s in self.signals if s.name == name), None)


def load_rubric(path: str | Path) -> Rubric:
    raw = Path(path).read_bytes()
    data: dict[str, Any] = yaml.safe_load(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return Rubric(**data, content_hash=digest)
