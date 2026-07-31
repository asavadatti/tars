# TARS — demo runbook

Six steps, about four minutes. Every command below was run against this repo;
the outputs are real.

## Before the session

```bash
cd tars
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_abcd.py          # full 10k corpus
cp .env.example .env                  # add your key
```

Pre-score a corpus so the demo reads from cache and never waits on the network.
This warms the cache, it does not fake anything: the key is
`(conversation_id, rubric_version, model_version)`, so any transcript the system
has not seen under that exact triple still goes to the model live.

```bash
export PYTHONPATH=src TARS_ABCD=data/abcd_v1.1.json
python -m tars.cli score --source abcd --limit 20
```

Twenty takes about four minutes and ~75k tokens against `claude-sonnet-4-6`;
the numbers in step 6 are from exactly this run. Two hundred is roughly forty
minutes, so start it well before the session or leave it at twenty — the demo
reads the same either way.

Start the server in a second terminal and leave it running:

```bash
export PYTHONPATH=src
uvicorn tars.api:app --port 8000
```

Fallback if the room's network is bad: prefix everything with `TARS_STUB=1`.
The stub judge is deterministic and needs no key, so the contract, the jobs,
the cache and the metrics all still demo. Say out loud that you are showing the
shape rather than the scores.

---

## 1. Show the contract before showing the output

```bash
curl -s localhost:8000/v1/rubric | jq
```

Open `localhost:8000/docs` alongside it. The point is that this is a documented
interface with a version on it, not a script that prints to a terminal.

Grab the rubric version, you need it in step 5:

```bash
RV=$(curl -s localhost:8000/v1/rubric | jq -r .version)   # default.v3+b4d83203
```

## 2. Submit a batch

```bash
curl -s -X POST localhost:8000/v1/score:batch \
  -H 'content-type: application/json' \
  -d '{"source":"abcd","limit":3}' | jq
```

```json
{
  "job_id": "job_4b97cb229f5b",
  "status": "running",
  "rubric_version": "default.v3+b4d83203",
  "model_version": "claude-sonnet-4-6",
  "items": [
    {"conversation_id": "abcd-3592", "status": "pending"},
    {"conversation_id": "abcd-9489", "status": "pending"},
    {"conversation_id": "abcd-3695", "status": "pending"}
  ]
}
```

Returns immediately with an id. Ten thousand transcripts do not fit in a
request/response cycle, and going from sync to async later breaks every caller.

## 3. Poll

```bash
curl -s localhost:8000/v1/jobs/job_4b97cb229f5b | jq '{status, items}'
```

```
status: succeeded
items: [(abcd-3592, done), (abcd-9489, done), (abcd-3695, done)]
```

Status is per item. Three failures in a batch of five hundred should not sink
the other four hundred and ninety seven.

## 4. Resubmit the identical batch

```bash
curl -s -X POST localhost:8000/v1/score:batch \
  -H 'content-type: application/json' \
  -d '{"source":"abcd","limit":3}' | jq -r .job_id
# then poll it
```

```
status: succeeded
items: [(abcd-3592, cached), (abcd-9489, cached), (abcd-3695, cached)]
```

Everything cached. The key is `(conversation_id, rubric_version, model_version)`.
This is what makes re-scoring cheap, and cheap re-scoring is what keeps the
rubric an editable file rather than a decision you are stuck with.

## 5. Pull one result and show the evidence

```bash
curl -s localhost:8000/v1/results/abcd-9489 | jq '.signals.goal_completion'
curl -s localhost:8000/v1/results/abcd-9489 | jq '.grounded'
```

```json
{
  "label": "completed",
  "confidence": 0.95,
  "abstained": false,
  "evidence": [
    {"turn_idx": 1, "quote": "just wanted to check on the status of a refund",
     "reason": "Customer's stated goal."},
    {"turn_idx": 17, "quote": "less than a week",
     "reason": "Agent provided the estimated timeline, completing the customer's inquiry."}
  ]
}
```

```json
[
  {
    "name": "abcd_action_completion",
    "passed": false,
    "detail": "executed 2/3 expected actions; missing ['notify-internal-team']"
  },
  {
    "name": "abcd_escalation_performed",
    "passed": false,
    "detail": "escalation required by procedure but never performed"
  }
]
```

This is the moment worth slowing down for, and on this conversation the two
kinds of check **disagree**. The model says `completed` with 0.95 confidence and
three quotes to back it. The arithmetic says the agent never fired
`notify-internal-team`, which this subflow's procedure requires. Both are
correct: the customer did get their answer, and the agent skipped a required
step. A single blended "quality score" would have hidden one of those.

That is the argument for the envelope. Judgement and measurement live side by
side, labelled differently, and the disagreement is visible instead of averaged
away.

Worth saying out loud on the escalation check: it fails in both directions.
Escalating nothing and escalating everything are different failures, and a
one-directional check would score the second as perfect.

Then show a conversation where the grounded check returns `null`:

```bash
curl -s localhost:8000/v1/results/abcd-3592 | jq -r '.grounded[0].detail'
# no expected action sequence mapped for subflow 'return_size'
```

The subflow names in `guidelines.json` do not all normalise onto the subflow
names in the conversation records. Coverage is partial, so the check abstains
instead of guessing.

## 5b. Score something they make up

The strongest liveness proof. Ask the panel for a two-line exchange, or use this:

```bash
curl -s -X POST localhost:8000/v1/score \
  -H 'content-type: application/json' \
  -d '{"turns":[
    {"speaker":"customer","text":"Third time I have contacted you about this charge. I am done."},
    {"speaker":"agent","text":"Please provide your order number."},
    {"speaker":"customer","text":"8812."},
    {"speaker":"agent","text":"I have refunded it. Anything else?"}
  ]}' | jq
```

Synchronous, because one conversation is one model call and the async ceremony
buys nothing. Batches stay async because that is the case that cannot be made
synchronous later without breaking callers.

The id is a content hash, so the same transcript twice is the same id and the
second run overwrites the first. Unlike the batch path, `/v1/score` does not
consult the cache before scoring — it always calls the model. That is what makes
this the liveness proof: nothing here can be served from a warmed cache. Pass
`conversation_id` explicitly if you want to control the id.

Note what abstains. On the transcript above, `goal_completion` has no stated
goal to judge against, and `procedure_adherence` has no documented procedure for
an ad-hoc request. `empathy_when_warranted` does fire here — the customer states
frustration in the first turn and the agent goes straight to process.
Abstention is the behaviour to point at, not a gap: signals declining to answer
is the system saying it has nothing falsifiable to say.

## 6. Filter, override, and show the metrics

```bash
curl -s 'localhost:8000/v1/results?signal=procedure_adherence&label=skipped_step' | jq -r '.count, (.results[].conversation_id)'
```

Returns four conversations from the cached twenty-three. `compliance_verification`
with `label=not_verified` returns exactly one, `abcd-4816`, if you want a second
filter to run.

Most other labels return nothing on a corpus this small, and that is worth
saying out loud rather than discovering live. A filter that legitimately returns
zero rows is not a broken filter.

Thresholds are applied at read time, not baked in at write time, so what counts
as bad stays changeable.

Disagree with one score. Four of the five signals are categorical, so the
correction is a label; `empathy_when_warranted` is ordinal and takes
`original_score`/`corrected_score` instead. The same endpoint handles both:

```bash
curl -s -X POST localhost:8000/v1/overrides \
  -H 'content-type: application/json' \
  -d "{\"conversation_id\":\"abcd-9489\",
       \"rubric_version\":\"$RV\",
       \"model_version\":\"claude-sonnet-4-6\",
       \"signal\":\"goal_completion\",
       \"original_label\":\"completed\",
       \"corrected_label\":\"partially_completed\",
       \"note\":\"customer got an answer, but the required internal notification never fired\",
       \"reviewer\":\"demo\"}"
```

Then:

```bash
curl -s localhost:8000/v1/metrics | jq
```

```
scored: 20   error_rate: 0.0
  goal_completion           abstention=0.000  conf=0.94
  procedure_adherence       abstention=0.450  conf=0.58
  escalation_judgment       abstention=0.150  conf=0.81
  empathy_when_warranted    abstention=0.700  conf=0.52
  compliance_verification   abstention=0.150  conf=0.81
agreement: {scored: 20, reviewed: 1, disagreements: 1, agreement_rate: 0.0}
```

Real numbers, `claude-sonnet-4-6` over 20 ABCD conversations, zero errors. Two
of them are worth volunteering before anyone asks.

`empathy_when_warranted` abstains 70% of the time. That is the conditional doing
its job — most support conversations contain no distress to acknowledge, and
without the `abstain_when` clause the mean would be an average over conversations
where empathy was never called for.

`procedure_adherence` has the lowest confidence in the set, 0.58, and abstains
45% of the time. That is the honest reading: ABCD conversations often have no
documented procedure to judge against, and the judge is telling you so rather
than guessing. It is the newest signal and the one whose criteria most needs
another pass.

One caveat if you re-run this: `/v1/metrics` filters on `rubric_version` but not
`model_version`, so stub results and live results under the same rubric are
averaged together. Score with `TARS_STUB=1` and these numbers will silently mix.

Close here. There is no accuracy number available because there is nothing to
be accurate against. Human agreement over time is the metric that exists, and
the override endpoint is what produces it.

---

## Optional: the MCP leg

If the session has time and the room is interested in the agent surface, this is
the strongest two minutes in the demo. In Claude Desktop, with the server
configured:

> Find conversations where the agent skipped a required procedure step, then
> show me the evidence for one of them.

Two tool calls, the second depending on the first. `find_conversations` returns
ids and scores only; `get_evidence` returns the quotes. Say why: an agent has no
scrollback and a finite context, so returning full evidence from a search would
consume the caller's context on the first query. Designing for a human developer
and designing for an agent consumer are different jobs.

---

## Things to have ready but not lead with

**The bug.** `is_scorable` originally checked only `score` and not `label`, so
both categorical signals reported 100% abstention. A plausible-looking number
that was completely wrong. Fixed in `schema.py` with a comment. Good answer to
"where did this surprise you."

**What you cut.** Auth, multi-tenancy, rate limiting, any trained classifier,
streaming, a UI, and voice. Voice is the interesting one: ASR output has no
sentence punctuation, imperfect diarization, and word error rates that
concentrate on the domain nouns the compliance check depends on.

**What is missing.** Repeat-run variance. The metrics endpoint reports how the
judge behaves but not whether a score of 2 is stable across runs. That is the
first thing to add and you should say so before anyone asks.
