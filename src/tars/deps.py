from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .judge import get_judge
from .jobs import JobRunner
from .rubric import load_rubric
from .store import Store

# src/tars/deps.py -> repo root. MCP clients launch the server with an arbitrary
# working directory, so relative paths in the environment must be anchored here
# rather than to cwd. Absolute paths are left alone.
ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Minimal .env loader.

    The README tells you to copy .env.example to .env, so that has to work
    without a dependency. Real environment variables always win, which is what
    makes the same code correct on a laptop and on a host that injects config.
    """
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()


def _resolve(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else ROOT / p)


@lru_cache(maxsize=1)
def context():
    rubric = load_rubric(_resolve(os.environ.get("TARS_RUBRIC", "rubrics/default.yaml")))
    store = Store(_resolve(os.environ.get("TARS_DB", "tars.db")))
    judge = get_judge(rubric, stub=os.environ.get("TARS_STUB") == "1")
    return rubric, store, judge, JobRunner(judge, rubric, store)


def find_conversation(source: str, conversation_id: str, path: str | None = None):
    """Locate a single conversation by id, or None.

    Linear scan, because the adapters expose an iterator over a file rather than
    an index. Building an index here would be inventing a storage layer the
    datasets do not have, and the corpus is re-read per call — fine for one
    lookup, the wrong shape for a hot path.
    """
    for convo in load_conversations(source, None, path):
        if convo.conversation_id == conversation_id:
            return convo
    return None


def load_conversations(source: str, limit: int | None, path: str | None = None):
    from .adapters import abcd, synthetic

    if source == "synthetic":
        return list(synthetic.load(limit))
    if source == "abcd":
        target = path or os.environ.get("TARS_ABCD", "data/abcd_sample.json")
        return list(abcd.load(_resolve(target), limit))
    raise ValueError(f"unknown source: {source}")
