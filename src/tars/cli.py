from __future__ import annotations

import argparse
import json
import sys
import time

from .deps import context, load_conversations


def main() -> None:
    p = argparse.ArgumentParser(prog="tars")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Score a batch and wait for it")
    s.add_argument("--source", default="synthetic", choices=["synthetic", "abcd"])
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--path", default=None)
    s.add_argument("--force", action="store_true")

    g = sub.add_parser("show", help="Print one result with evidence")
    g.add_argument("conversation_id")

    sub.add_parser("metrics", help="Abstention, confidence, error and agreement rates")

    args = p.parse_args()
    rubric, store, judge, runner = context()

    if args.cmd == "score":
        convos = load_conversations(args.source, args.limit, args.path)
        job = runner.submit(convos, force=args.force)
        print(f"job {job.job_id}  rubric={rubric.version}  model={judge.model_version}")
        while runner.get(job.job_id).status in ("queued", "running"):
            time.sleep(0.4)
        final = runner.get(job.job_id)
        print(f"{final.status}: {final.counts}")

    elif args.cmd == "show":
        result = store.get(args.conversation_id, rubric.version, judge.model_version)
        print(result.model_dump_json(indent=2) if result else "not scored")

    elif args.cmd == "metrics":
        from .api import metrics
        print(json.dumps(metrics(), indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Piping to head/less closes stdout early. Exiting quietly beats
        # printing a traceback over the thing you were trying to read.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
