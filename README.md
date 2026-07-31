# TARS

**Transcript Analysis and Review Service.** A conversation quality reviewer.

Summary: Scores human-human customer-support transcripts against a set of customizable metrics using LLM. Returns structured quality signals with the transcript evidence behind each one

Endpoints [CLI, REST, MCP:
score: GET: /v1/score
Score one or more conversations by passing in the conversation text or the canonical conversation id. You can also rescore a previous conversation (if the model or rubric has changed for example).

You can supply the `conversation_id` to score the exact conversation or add a new conversation inline to score. The inline conversation must be in the correct JSON format.

Query Params: 
include string all|signals
all: Returns the entire JSON response
signals: Returns only the quality signals



## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=src

# No API key needed. Runs the whole pipeline against a deterministic fake judge.
TARS_STUB=1 python -m tars.cli score --source abcd --limit 3
TARS_STUB=1 python -m tars.cli show abcd-3592
TARS_STUB=1 python -m tars.cli metrics

# Against the real model
cp .env.example .env      # add your key
python -m tars.cli score --source abcd --limit 20
uvicorn tars.api:app --reload     # docs at /docs
```

Three ABCD conversations ship in `data/`. For the full 10k, run
`python scripts/fetch_abcd.py` and set `TARS_ABCD=data/abcd_v1.1.json`.

## The contract

```
POST /v1/score            score one transcript synchronously
POST /v1/score:batch      submit, returns a job id immediately
GET  /v1/jobs/{id}        poll; per-item status
GET  /v1/results/{id}     one result, with evidence
GET  /v1/results          filter by signal and threshold
POST /v1/overrides        record a human correction
GET  /v1/metrics          abstention, confidence, error and agreement rates
GET  /v1/rubric           active rubric and its version
```

A result looks like this:

```json
{
  "conversation_id": "abcd-3592",
  "rubric_version": "default.v3+b4d83203",
  "model_version": "claude-sonnet-4-6",
  "signals": {
    "empathy_when_warranted": {
      "score": 2.0,
      "confidence": 0.81,
      "abstained": false,
      "evidence": [
        {"turn_idx": 7, "quote": "Please provide your order number.",
         "reason": "customer had just said this was their third contact; agent moved straight to process"}
      ],
      "rationale": "Frustration was stated explicitly and never acknowledged."
    }
  },
  "grounded": [
    {"name": "abcd_action_completion", "passed": false,
     "detail": "executed 2/3 expected actions; missing ['notify-internal-team']"},
    {"name": "abcd_escalation_performed", "passed": false,
     "detail": "escalation required by procedure but never performed"}
  ]
}
```

The same core is exposed over MCP (`python -m tars.mcp_server`) with six tools:
`list_signals`, `score_conversations`, `score_transcript`, `check_job`,
`find_conversations`, `get_evidence`. `score_conversations` returns a job id and
`check_job` polls it, mirroring the async batch contract; `score_transcript`
scores a single inline transcript and returns the result directly.
The agent-facing tools return ids and one-line summaries; evidence is a separate
call, because returning full evidence from a search would consume the caller's
context on the first query.

## Signals

Five, in `rubrics/default.yaml`. They target an AI agent rather than a human one:
two ask what the conversation achieved, three describe how the agent behaved
getting there. Each was chosen to demonstrate a different property, not to be
commercially complete.

`goal_completion` asks only whether the customer's stated reason got resolved. It
carries a deterministic cross-check: ABCD ships the action sequence the agent
actually executed, and `data/guidelines.json` lists the sequence each subflow is
supposed to follow. Comparing them is arithmetic, not judgement. Coverage is
partial — subflow names in the guidelines file do not all normalise onto the
subflow names in the conversation records. `refund_status` matches, `return_size`
does not. Where no mapping exists the check returns `passed: null` rather than a
guess.

`procedure_adherence` is scoped to *how*, not *whether*, which is the reason it
is separate from `goal_completion`. An agent that resolves the problem by
improvising past the procedure passes one and fails the other, and collapsing
them would hide exactly that case. Its labels name distinct failure modes —
`skipped_step`, `unauthorized_step` — instead of grading severity, because
"minor deviation" is not falsifiable and "omitted a required step" is.

`escalation_judgment` folds three questions into one signal: were the right
things escalated, only those things, and soon enough. It is the second signal
with a deterministic cross-check. `abcd_escalation_performed` compares whether
the subflow's procedure calls for `notify-internal-team` against whether that
action fired, and fails in *both* directions — a one-directional check would
score an agent that escalates everything as perfect. Timeliness stays
judgement, because nothing in the data pins down when a hand-off became due.

`empathy_when_warranted` abstains when the customer expressed no difficulty.
Without that condition the aggregate is dominated by neutral conversations where
empathy was never called for, and the number stops meaning anything. It scores
the acknowledgement visible in the transcript, not whether it reads as sincere —
a transcript cannot show whether an agent, human or model, meant it, and a
signal that claims otherwise is not falsifiable.

`compliance_verification` is deliberately narrow: was identity confirmed before
an account-modifying action. A broad "compliance risk" score is unfalsifiable. A
single named check is auditable, and a customer can configure it.

Cut, and worth saying why: `pressure_resistance` — whether the customer can argue
the agent out of a refusal, a policy, or a verification step. It is the signal
that matters most for an autonomous agent and the one this corpus cannot test,
because ABCD is human-to-human support where that pattern is rare. Reporting an
abstention rate that only describes the dataset would be worse than not shipping
it. It belongs against real agent traffic.

A note on the corpus generally: `escalation_judgment` also abstains more often on
ABCD than it would on real agent traffic, since only some subflows map onto a
documented action sequence. That is the honest result, not a gap to paper over.

## What is expensive to change

The result envelope, once anything reads it. Adding fields is free; renaming is
not.

The provenance tuple. A score carrying no `rubric_version` and `model_version` is
uninterpretable and cannot be backfilled. The store's primary key is
`(conversation_id, rubric_version, model_version)` so that this stays visible
rather than becoming a convention people forget.

Mandatory evidence. If evidence were optional, some share of records would never
have it, and those records could never be audited.

Async job semantics. Sync to async breaks every caller; async to sync is trivial.
The demo does not need a job queue and production cannot work without one.

Scoring at conversation grain with turn-indexed evidence. Turn-level scoring
cannot be recovered later without re-running everything, and evidence pointers
give attribution without paying for it.

## What is cheap to change

The model, behind `judge.Judge`. Prompt wording, since every result is
version-stamped and re-runnable. Storage, behind `store.Store`. Flagging
thresholds, which are applied at read time in `store.query` rather than baked in
at write time. Adding an adapter, which touches nothing downstream.

And the signals themselves. Which five metrics to compute is the most contestable
decision here and the one least likely to survive contact with a real QA team, so
the rubric is a YAML file rather than code. That only works if re-scoring is
cheap, which is what the idempotency key on
`(conversation_id, rubric_version, model_version)` buys: re-running a corpus under
an edited rubric is a cache miss on exactly the records that changed.

The rubric version includes a content hash of the file, so editing criteria text
without bumping the revision still produces a new version. Otherwise a trendline
drawn across the edit would be silently wrong.

## Deliberately not built

Auth, multi-tenancy, rate limiting. Fine-tuning or any trained classifier.
Streaming or real-time scoring. A UI. Voice transcripts, which are the input this
would degrade on hardest: ASR output has no sentence punctuation, imperfect
diarization, and word error rates that concentrate on the domain nouns the
compliance check depends on.

Judge reliability is measured but not yet established. `/v1/metrics` reports
abstention and confidence, which describe the judge's behaviour. It does not
report variance across repeated runs of the same conversation, which is what
would tell you whether a score of 2 is stable. That is the first thing to add.

## Tests

```bash
TARS_STUB=1 PYTHONPATH=src python -m pytest tests -q
```

Nine tests. `test_escalation_check_catches_both_directions` pins down the
two-directional property above, and
`test_categorical_override_counts_as_a_disagreement` covers a rubric of purely
categorical signals, where comparing only numeric scores made every human
correction register as agreement. The one worth reading is
`test_rubric_version_changes_when_criteria_change`. The one that came from a real
bug is `test_error_result_is_not_a_cache_hit`: a stored failure used to count as
a cache hit, so a single transient 401 pinned a conversation to that error until
the row was deleted by hand.
