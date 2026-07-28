"""Batch orchestration.

Async from the first commit. Sync to async is a breaking change for every
caller; async to sync is trivial. The demo does not need this and production
cannot work without it, so it costs an hour now and a migration later.

In-process executor on purpose. Swapping in Celery or arq touches this file only.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .grounding import run_checks
from .judge import Judge
from .rubric import Rubric
from .schema import Conversation, Job, JobItem
from .store import Store


class JobRunner:
    def __init__(self, judge: Judge, rubric: Rubric, store: Store, workers: int = 4):
        self._judge, self._rubric, self._store = judge, rubric, store
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._jobs: dict[str, Job] = {}

    def submit(self, conversations: list[Conversation], force: bool = False) -> Job:
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            rubric_version=self._rubric.version,
            model_version=self._judge.model_version,
            items=[JobItem(conversation_id=c.conversation_id) for c in conversations],
        )
        self._jobs[job.job_id] = job
        self._pool.submit(self._run, job, conversations, force)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _run(self, job: Job, conversations: list[Conversation], force: bool) -> None:
        job.status = "running"
        by_id = {i.conversation_id: i for i in job.items}
        grounded_by = [s.grounded_by for s in self._rubric.signals if s.grounded_by]

        for convo in conversations:
            item = by_id[convo.conversation_id]
            try:
                # Idempotency: same conversation, same rubric, same model is a
                # cache hit. This is what keeps re-scoring cheap, and cheap
                # re-scoring is what keeps the rubric a reversible decision.
                if not force:
                    cached = self._store.get(convo.conversation_id, job.rubric_version, job.model_version)
                    # A stored failure is not a cache hit. Errors are still
                    # persisted, because /v1/metrics derives error_rate from
                    # them, but they must never suppress a retry: one expired
                    # key would otherwise pin a conversation to a 401 forever.
                    # The retry's INSERT OR REPLACE overwrites the error row.
                    if cached is not None and not cached.error:
                        item.status = "cached"
                        continue

                result = self._judge.score(convo, self._rubric)
                result.grounded = run_checks(convo, grounded_by)
                self._store.put(result)
                item.status = "error" if result.error else "done"
                item.error = result.error
            except Exception as exc:  # noqa: BLE001 - one bad item must not sink the batch
                item.status = "error"
                item.error = f"{type(exc).__name__}: {exc}"

        counts = job.counts
        job.status = "failed" if counts.get("error", 0) == len(job.items) else (
            "partial" if counts.get("error") else "succeeded"
        )
        job.finished_at = datetime.now(timezone.utc)
