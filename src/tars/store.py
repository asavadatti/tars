"""Persistence behind a narrow interface.

SQLite here, Postgres later, and the swap costs nothing because everything above
this line only knows the four methods below. The idempotency key is the frozen
part: (conversation_id, rubric_version, model_version).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import Override, ScoredConversation

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
  conversation_id TEXT NOT NULL,
  rubric_version  TEXT NOT NULL,
  model_version   TEXT NOT NULL,
  source          TEXT NOT NULL,
  scored_at       TEXT NOT NULL,
  payload         TEXT NOT NULL,
  PRIMARY KEY (conversation_id, rubric_version, model_version)
);
CREATE INDEX IF NOT EXISTS idx_results_rubric ON results(rubric_version, model_version);

CREATE TABLE IF NOT EXISTS overrides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id TEXT NOT NULL,
  rubric_version  TEXT NOT NULL,
  model_version   TEXT NOT NULL,
  signal          TEXT NOT NULL,
  payload         TEXT NOT NULL,
  created_at      TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str = "tars.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, result: ScoredConversation) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?)",
            (result.conversation_id, result.rubric_version, result.model_version,
             result.source, result.scored_at.isoformat(), result.model_dump_json()),
        )
        self._conn.commit()

    def get(self, conversation_id: str, rubric_version: str, model_version: str) -> ScoredConversation | None:
        row = self._conn.execute(
            "SELECT payload FROM results WHERE conversation_id=? AND rubric_version=? AND model_version=?",
            (conversation_id, rubric_version, model_version),
        ).fetchone()
        return ScoredConversation(**json.loads(row["payload"])) if row else None

    def query(self, rubric_version: str | None = None, signal: str | None = None,
              max_score: float | None = None, label: str | None = None,
              limit: int = 50) -> list[ScoredConversation]:
        """Filtering happens here, not at write time.

        Thresholds are a read-time concern on purpose. Store raw scores, derive
        flags on query, and 'what counts as bad' stays changeable forever.
        """
        sql = "SELECT payload FROM results"
        params: list[object] = []
        if rubric_version:
            sql += " WHERE rubric_version=?"
            params.append(rubric_version)
        sql += " ORDER BY scored_at DESC LIMIT ?"
        params.append(limit * 5)

        out = []
        for row in self._conn.execute(sql, params):
            r = ScoredConversation(**json.loads(row["payload"]))
            if signal:
                sig = r.signals.get(signal)
                if sig is None or sig.abstained:
                    continue
                if max_score is not None and (sig.score is None or sig.score > max_score):
                    continue
                if label is not None and sig.label != label:
                    continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def add_override(self, override: Override) -> None:
        self._conn.execute(
            "INSERT INTO overrides (conversation_id, rubric_version, model_version, signal, payload, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (override.conversation_id, override.rubric_version, override.model_version,
             override.signal, override.model_dump_json(), override.created_at.isoformat()),
        )
        self._conn.commit()

    def agreement_rate(self, rubric_version: str) -> dict[str, float | int]:
        """Human-agreement over time is the metric that says whether this works.

        Accuracy against nothing is not available. Disagreement rate is.
        """
        rows = self._conn.execute(
            "SELECT payload FROM overrides WHERE rubric_version=?", (rubric_version,)
        ).fetchall()
        total = self._conn.execute(
            "SELECT COUNT(*) c FROM results WHERE rubric_version=?", (rubric_version,)
        ).fetchone()["c"]
        def _corrected(payload: str) -> bool:
            p = json.loads(payload)
            # Ordinal signals carry scores, categorical ones carry labels.
            # Comparing only scores reported every categorical correction as
            # agreement, which is the same shape of bug as SignalResult
            # .is_scorable. .get() keeps rows written before labels existed.
            before = p["original_score"] if p["original_score"] is not None else p.get("original_label")
            after = p["corrected_score"] if p["corrected_score"] is not None else p.get("corrected_label")
            return before != after

        changed = sum(1 for r in rows if _corrected(r["payload"]))
        return {"scored": total, "reviewed": len(rows), "disagreements": changed,
                "agreement_rate": round(1 - changed / len(rows), 3) if rows else -1.0}
