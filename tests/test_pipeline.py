import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["TARS_STUB"] = "1"
# Point the store at a throwaway db before anything imports tars.deps, which
# caches the context on first use. The inline-scoring tests go through the real
# context(), so without this every test run writes stub rows into the tars.db
# the demo reads from — and /v1/metrics averages across model_version, so those
# rows silently skew live numbers.
os.environ["TARS_DB"] = os.path.join(tempfile.mkdtemp(prefix="tars-test-"), "test.db")

from tars.adapters import abcd, synthetic
from tars.grounding import abcd_action_completion
from tars.rubric import load_rubric
from tars.schema import Speaker


def test_abcd_adapter_produces_contiguous_turn_indices():
    convo = next(abcd.load("data/abcd_sample.json", limit=1))
    assert [t.idx for t in convo.turns] == list(range(len(convo.turns)))
    assert convo.source_metadata["subflow"]
    assert any(t.speaker is Speaker.CUSTOMER for t in convo.turns)


def test_grounding_returns_unknown_rather_than_guessing():
    for convo in abcd.load("data/abcd_sample.json"):
        check = abcd_action_completion(convo)
        assert check.passed in (True, False, None)
        if check.passed is None:
            assert "no expected action sequence" in check.detail


def test_escalation_check_catches_both_directions(tmp_path):
    """A one-directional check would score an escalate-everything agent perfectly.

    'refund_status' requires notify-internal-team; 'initiate_refund' does not.
    Both misses and gratuitous hand-offs must fail, and an unmapped subflow must
    abstain rather than guess.
    """
    from tars.grounding import abcd_escalation_performed
    from tars.schema import Conversation, Turn

    def convo(subflow, executed):
        return Conversation(
            conversation_id="x", source="abcd",
            turns=[Turn(idx=0, speaker=Speaker.CUSTOMER, text="hi")],
            source_metadata={"subflow": subflow, "executed_actions": executed},
        )

    required_and_done = abcd_escalation_performed(
        convo("refund_status", ["pull-up-account", "notify-internal-team"]))
    required_and_missed = abcd_escalation_performed(
        convo("refund_status", ["pull-up-account"]))
    not_required_but_done = abcd_escalation_performed(
        convo("initiate_refund", ["pull-up-account", "notify-internal-team"]))
    not_required_none = abcd_escalation_performed(
        convo("initiate_refund", ["pull-up-account"]))
    unmapped = abcd_escalation_performed(convo("return_size", []))

    assert required_and_done.passed is True
    assert required_and_missed.passed is False
    assert not_required_but_done.passed is False, "over-escalation must not pass"
    assert not_required_none.passed is True
    assert unmapped.passed is None
    assert "no expected action sequence" in unmapped.detail


def test_rubric_version_changes_when_criteria_change(tmp_path):
    a = load_rubric("rubrics/default.yaml")
    edited = tmp_path / "r.yaml"
    edited.write_text(open("rubrics/default.yaml").read().replace("Judge the outcome", "Judge outcomes"))
    b = load_rubric(str(edited))
    assert a.version != b.version, "content hash must catch silent criteria edits"


def test_synthetic_fixtures_load():
    assert len(list(synthetic.load())) == 4


def test_inline_scoring_accepts_turns_without_idx():
    """Regression: the documented payload omits idx and must not 422."""
    from tars.api import InlineRequest, score_inline

    r = score_inline(InlineRequest(turns=[
        {"speaker": "customer", "text": "Charged twice, third time asking."},
        {"speaker": "agent", "text": "Order number?"},
    ]))
    assert r.conversation_id.startswith("inline-")
    assert r.source == "inline"
    assert set(r.signals) == {"goal_completion", "procedure_adherence",
                              "escalation_judgment", "empathy_when_warranted",
                              "compliance_verification"}


def test_inline_id_is_stable_for_identical_content():
    from tars.api import InlineRequest, score_inline

    payload = [{"speaker": "customer", "text": "hi"}, {"speaker": "agent", "text": "hello"}]
    a = score_inline(InlineRequest(turns=payload))
    b = score_inline(InlineRequest(turns=payload))
    assert a.conversation_id == b.conversation_id


def test_scoring_by_id_runs_the_same_grounding_as_the_batch_path():
    """Regression: /v1/score used to skip grounding entirely.

    Scoring a corpus conversation by id writes to the same
    (conversation, rubric, model) row the batch path writes. If this route
    omitted the deterministic checks, scoring an already-batched conversation
    would silently replace a complete record with a poorer one.
    """
    from tars.api import InlineRequest, score_inline
    from tars.deps import find_conversation

    convo = find_conversation("synthetic", "synthetic-unverified_refund")
    assert convo is not None, "find_conversation must resolve a known fixture id"

    r = score_inline(InlineRequest(conversation_id=convo.conversation_id, source="synthetic",
                                   force=True))
    assert [g.name for g in r.grounded] == ["abcd_action_completion", "abcd_escalation_performed"]
    assert all(g.passed is None for g in r.grounded), "non-ABCD source must abstain, not guess"


def test_find_conversation_returns_none_for_unknown_id():
    from tars.deps import find_conversation

    assert find_conversation("synthetic", "synthetic-does-not-exist") is None


def test_signal_projection_keeps_error_and_provenance():
    """Regression: a projected result must never be less interpretable.

    Failed runs are persisted with an empty `signals` map, so a projection that
    dropped `error` would answer 200 with `{}` and give the caller no way to
    tell "scored nothing" from "the key was rejected". Provenance goes the same
    way — a score with no rubric or model attached cannot be compared later.
    """
    from tars.api import ResultSummary
    from tars.schema import ScoredConversation, SignalResult

    failed = ResultSummary.of(ScoredConversation(
        conversation_id="c2", source="abcd", rubric_version="r1", model_version="m1",
        error="AuthenticationError: 401 invalid x-api-key"))
    assert failed.signals == {}
    assert failed.error == "AuthenticationError: 401 invalid x-api-key"
    assert (failed.rubric_version, failed.model_version) == ("r1", "m1")

    scored = ResultSummary.of(ScoredConversation(
        conversation_id="c1", source="abcd", rubric_version="r1", model_version="m1",
        signals={
            "cat": SignalResult(name="cat", label="completed", confidence=0.9),
            "ord": SignalResult(name="ord", score=2.0, confidence=0.8),
            "abs": SignalResult(name="abs", confidence=0.0, abstained=True),
        }))
    assert scored.signals["cat"].value == "completed", "categorical carries its label"
    assert scored.signals["ord"].value == 2.0, "ordinal carries its score"
    assert scored.signals["abs"].value is None and scored.signals["abs"].abstained
    assert scored.error is None


def test_categorical_override_counts_as_a_disagreement(tmp_path):
    """Regression: every signal in the default rubric is categorical.

    agreement_rate compared only scores, so a reviewer flipping a label
    registered as agreement and the metric read 1.0 however much they disagreed.
    """
    from tars.schema import Override
    from tars.store import Store

    store = Store(str(tmp_path / "t.db"))
    rv = "default.v2+testing"
    store.add_override(Override(
        conversation_id="c1", rubric_version=rv, model_version="m",
        signal="escalation_judgment",
        original_label="appropriate", corrected_label="missed_escalation", reviewer="test"))
    store.add_override(Override(
        conversation_id="c2", rubric_version=rv, model_version="m",
        signal="escalation_judgment",
        original_label="appropriate", corrected_label="appropriate", reviewer="test"))

    a = store.agreement_rate(rv)
    assert a["reviewed"] == 2
    assert a["disagreements"] == 1, "a flipped label must count as a disagreement"
    assert a["agreement_rate"] == 0.5


def test_error_result_is_not_a_cache_hit(tmp_path):
    """Regression: a stored failure must never suppress a retry.

    Errors are persisted, because /v1/metrics derives error_rate from them, but
    counting one as a cache hit pinned a conversation to a transient 401 for as
    long as the row survived. Re-scoring is the whole point of the cache key.
    """
    import time

    from tars.jobs import JobRunner
    from tars.judge import StubJudge
    from tars.schema import Conversation, ScoredConversation, Turn
    from tars.store import Store

    rubric = load_rubric("rubrics/default.yaml")
    convo = Conversation(conversation_id="probe", source="inline", turns=[
        Turn(idx=0, speaker=Speaker.CUSTOMER, text="Charged twice, third time asking."),
        Turn(idx=1, speaker=Speaker.AGENT, text="Order number?"),
    ])
    calls: list[int] = []

    class FailsOnce(StubJudge):
        def score(self, convo, rubric):
            calls.append(1)
            if len(calls) == 1:
                return ScoredConversation(
                    conversation_id=convo.conversation_id, source=convo.source,
                    rubric_version=rubric.version, model_version=self.model_version,
                    error="AuthenticationError: 401 invalid x-api-key")
            return super().score(convo, rubric)

    runner = JobRunner(FailsOnce(rubric), rubric, Store(str(tmp_path / "t.db")))

    def run() -> str:
        job = runner.submit([convo])
        deadline = time.time() + 10
        while job.finished_at is None and time.time() < deadline:
            time.sleep(0.01)
        assert job.finished_at is not None, "job did not finish"
        return job.items[0].status

    assert run() == "error"
    assert run() == "done", "a stored error must be retried, not served from cache"
    assert run() == "cached", "a successful result must still cache"
    assert len(calls) == 2, "the cached run must not reach the judge"
