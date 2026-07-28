"""ABCD adapter.

Verified against data/abcd_sample.json from asappresearch/abcd. Shape:

    {
      "convo_id": 3592,
      "scenario": {"flow": "product_defect", "subflow": "return_size", ...},
      "original": [["agent", "Hi!"], ["customer", "..."], ["action", "..."]],
      "delexed": [{"speaker", "text", "turn_count", "targets", "candidates"}]
    }

`targets` is positional: [subflow, task, action_name, slot_values, utt_id].
Action rows carry task == "take_action" and the action name at index 2.
That sequence is the ground truth the goal_completion signal is checked against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..schema import Conversation, Speaker, Turn

SOURCE = "abcd"

# ABCD marks system-executed actions with the pseudo-speaker "action". They are
# not utterances. Keeping them as SYSTEM turns preserves ordering for evidence
# indices; the judge prompt can be told to ignore them.
_SPEAKER_MAP = {
    "customer": Speaker.CUSTOMER,
    "agent": Speaker.AGENT,
    "action": Speaker.SYSTEM,
}


def _executed_actions(record: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for turn in record.get("delexed", []):
        targets = turn.get("targets") or []
        if len(targets) >= 3 and targets[1] == "take_action" and targets[2]:
            out.append(targets[2])
    return out


def to_canonical(record: dict[str, Any]) -> Conversation:
    turns = [
        Turn(idx=i, speaker=_SPEAKER_MAP.get(speaker, Speaker.SYSTEM), text=text)
        for i, (speaker, text) in enumerate(record["original"])
    ]
    scenario = record.get("scenario", {})
    return Conversation(
        conversation_id=f"abcd-{record['convo_id']}",
        source=SOURCE,
        turns=turns,
        source_metadata={
            "flow": scenario.get("flow"),
            "subflow": scenario.get("subflow"),
            "executed_actions": _executed_actions(record),
        },
    )


def load(path: str | Path, limit: int | None = None, split: str = "train") -> Iterator[Conversation]:
    """Reads either abcd_sample.json (a bare list) or abcd_v1.1.json (split dict)."""
    data = json.loads(Path(path).read_text())
    records = data if isinstance(data, list) else data.get(split, [])
    for i, record in enumerate(records):
        if limit is not None and i >= limit:
            return
        yield to_canonical(record)
