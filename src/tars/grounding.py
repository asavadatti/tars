"""Deterministic checks that do not involve a model.

Exactly one signal in the default rubric has a verifiable answer. Keeping that
check in its own module, with its own failure mode, is the point: it makes the
boundary between "measured" and "judged" explicit rather than rhetorical.

Coverage is partial and that is reported, not hidden. Subflow names in
guidelines.json do not all normalise onto the subflow names in the conversation
records (refund_status lines up, return_size does not). Where no expected
sequence exists, the check returns passed=None. It never guesses.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schema import Conversation, GroundedCheck

_IGNORED_BUTTONS = {"n/a", "na", ""}


@lru_cache(maxsize=1)
def _expected_actions(guidelines_path: str) -> dict[str, list[str]]:
    path = Path(guidelines_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, list[str]] = {}
    for flow in data.values():
        for subflow_name, spec in flow.get("subflows", {}).items():
            key = subflow_name.lower().replace(" ", "_")
            buttons = [
                a["button"].lower().replace(" ", "-")
                for a in spec.get("actions", [])
                if a.get("button", "").lower() not in _IGNORED_BUTTONS
            ]
            if buttons:
                out[key] = buttons
    return out


def abcd_action_completion(
    convo: Conversation, guidelines_path: str = ""
) -> GroundedCheck:
    from .deps import ROOT

    guidelines_path = guidelines_path or str(ROOT / "data" / "guidelines.json")
    name = "abcd_action_completion"

    if convo.source != "abcd":
        return GroundedCheck(name=name, passed=None, detail="not an ABCD conversation")

    subflow = convo.source_metadata.get("subflow")
    expected = _expected_actions(guidelines_path).get(subflow or "")
    if not expected:
        return GroundedCheck(
            name=name,
            passed=None,
            detail=f"no expected action sequence mapped for subflow '{subflow}'",
        )

    executed = set(convo.source_metadata.get("executed_actions", []))
    missing = [a for a in expected if a not in executed]

    return GroundedCheck(
        name=name,
        passed=not missing,
        detail=(
            f"executed {len(expected) - len(missing)}/{len(expected)} expected actions"
            + (f"; missing {missing}" if missing else "")
        ),
    )


_ESCALATION_ACTION = "notify-internal-team"


def abcd_escalation_performed(
    convo: Conversation, guidelines_path: str = ""
) -> GroundedCheck:
    """Did the hand-off fire when the procedure called for one, and only then?

    Deterministic half of escalation_judgment. Whether escalation was *warranted*
    and whether it was *timely* are judgement; whether the documented procedure
    required it and whether the action actually fired are both recorded facts.

    Checks both directions on purpose — a missed hand-off and a gratuitous one
    are different failures, and an agent that escalates everything would score
    perfectly against a one-directional check.
    """
    from .deps import ROOT

    guidelines_path = guidelines_path or str(ROOT / "data" / "guidelines.json")
    name = "abcd_escalation_performed"

    if convo.source != "abcd":
        return GroundedCheck(name=name, passed=None, detail="not an ABCD conversation")

    subflow = convo.source_metadata.get("subflow")
    expected = _expected_actions(guidelines_path).get(subflow or "")
    if not expected:
        return GroundedCheck(
            name=name,
            passed=None,
            detail=f"no expected action sequence mapped for subflow '{subflow}'",
        )

    required = _ESCALATION_ACTION in expected
    performed = _ESCALATION_ACTION in set(convo.source_metadata.get("executed_actions", []))

    if required and performed:
        detail = "escalation required by procedure and performed"
    elif required and not performed:
        detail = "escalation required by procedure but never performed"
    elif performed:
        detail = "escalated although this subflow's procedure does not call for it"
    else:
        detail = "no escalation required and none performed"

    return GroundedCheck(name=name, passed=required == performed, detail=detail)


CHECKS = {
    "abcd_action_completion": abcd_action_completion,
    "abcd_escalation_performed": abcd_escalation_performed,
}


def run_checks(convo: Conversation, names: list[str]) -> list[GroundedCheck]:
    return [CHECKS[n](convo) for n in names if n in CHECKS]
