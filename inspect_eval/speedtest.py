"""Where a run's time goes, read from Inspect logs. Makes no model calls.

    python inspect_eval/speedtest.py logs/<run>.eval [more.eval ...]
    python inspect_eval/speedtest.py --same-prompts before.eval after.eval
    python inspect_eval/speedtest.py --same-labels before.eval after.eval

The first form prints, per log, the run's wall clock and the model-under-test calls split
by single- vs multi-turn item and by opening vs pressure turn: calls, failed calls and
retries, tokens in and out, turns that hit ``max_tokens``, empty answers (no reply text,
e.g. when reasoning used the whole budget), latency, and tokens per second. Each call's time is split into
``generate`` (the request to the provider, as Inspect times it in ``output.time``) and
``wait`` (time the call spent inside Inspect before and around that request, mostly
queueing for a ``--max-connections`` slot). Queueing inside the server, e.g. Ollama
holding a request until a parallel slot frees up, is part of ``generate``; compare
tokens/s at concurrency 1 to see it. Judge calls (model role ``grader``) are reported on
their own line, so a run with ``--no-score`` measures the solver alone.

``retries`` is what the provider reported in ``ModelEvent.retries``; not every provider
sets it. A call that errored and was retried by Inspect always shows in ``failed_calls``.
``slowest_samples`` lists the five samples with the longest total time. A log with no
stored samples, or a run that did not finish, gets a ``warnings`` entry, also printed to
stderr. Without stored samples, ``samples`` and ``errors`` still come from Inspect's own
totals (as in ``report.py``), and every figure that needs the samples is ``null``, not 0.

``--same-prompts`` checks the solver's ground rule for a speed change: the model sees the
same thing. Replies are sampled, so a later turn's prompt differs between runs by the
model's own earlier replies. It therefore compares, per sample, epoch and turn, the HTTP
request body sent to the provider byte for byte (generation settings, message order, every
system and user message), after checking that each assistant message in it is exactly the
reply the model gave earlier in the same run. It exits non-zero on any difference.

``--same-labels`` is the scorer's version, for two scored logs of the same transcripts
(``inspect score`` on one solver log, before and after a scorer change). Per sample and
epoch it compares the score value, i.e. every turn's label, and the text of every prompt
sent to the judge. It prints each difference, then the label agreement, and exits
non-zero if anything differs. A change that shouldn't touch grading (concurrency,
caching) must pass; for one that might (an output cap, a different judge) the agreement
is the number to report.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime

from inspect_ai.log import read_eval_log

from report import coverage


def model_calls(sample, role=None, failed=False):
    """The sample's model events for one model role (None: the model under test).

    A call that times out and is retried leaves an extra event carrying the error; those
    are left out unless ``failed``, so the n-th event is the n-th turn.
    """
    return [
        e for e in sample.events
        if e.event == "model" and e.role == role and (failed or e.error is None)
    ]


def call_row(event):
    usage = event.output.usage
    wall = (event.completed - event.timestamp).total_seconds() if event.completed else None
    gen = event.output.time
    return {
        "in": usage.input_tokens if usage else 0,
        "out": usage.output_tokens if usage else 0,
        "reasoning": (usage.reasoning_tokens or 0) if usage else 0,
        "wall": wall,
        "generate": gen,
        "wait": None if wall is None or gen is None else max(wall - gen, 0.0),
        "stop": event.output.stop_reason,
        "retries": event.retries or 0,
        # A call that ended in an error (e.g. "Request timed out.") with no output.
        "failed": event.error is not None,
        # The reply has no answer text, e.g. all max_tokens went to reasoning.
        "empty": event.error is None and not event.output.completion.strip(),
    }


def summarize(rows):
    def pct(values, q):
        values = sorted(v for v in values if v is not None)
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    gen = [r["generate"] for r in rows if r["generate"]]
    out = sum(r["out"] for r in rows)
    return {
        "calls": len(rows),
        "failed_calls": sum(r["failed"] for r in rows),
        "retries": sum(r["retries"] for r in rows),
        "empty_answers": sum(r["empty"] for r in rows),
        "tokens_in": sum(r["in"] for r in rows),
        "tokens_out": out,
        "reasoning_tokens": sum(r["reasoning"] for r in rows),
        "hit_max_tokens": sum(r["stop"] == "max_tokens" for r in rows),
        "generate_s": round(sum(gen), 1),
        "wait_s": round(sum(r["wait"] or 0 for r in rows), 1),
        "latency_p50_s": pct([r["wall"] for r in rows], 0.5),
        "latency_p95_s": pct([r["wall"] for r in rows], 0.95),
        # Per-request decode speed, comparable across concurrency settings only as a
        # measure of how much each request slows when the server shares its GPU.
        "out_tok_per_s_per_call": round(out / sum(gen), 1) if gen else None,
    }


def report(path):
    log = read_eval_log(path, resolve_attachments=True)
    samples = log.samples or []
    warnings = []
    if not samples:
        warnings.append("log has no samples (run with --no-log-samples, or stopped before any finished): nothing to time")
    if log.status != "success":
        warnings.append(f"run status is {log.status!r}: numbers cover only what the log recorded")
    start, end = log.stats.started_at, log.stats.completed_at
    wall = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() if start and end else None
    groups, judge = defaultdict(list), []
    for sample in samples:
        kind = "multi-turn" if sample.metadata.get("multiturn") else "single-turn"
        turn = 0
        for event in model_calls(sample, failed=True):
            groups[(kind, "opening" if turn == 0 else "pressure")].append(call_row(event))
            turn += event.error is None
        judge += [call_row(e) for e in model_calls(sample, "grader", failed=True)]
    solver_rows = [r for rows in groups.values() for r in rows]
    out_tokens = sum(r["out"] for r in solver_rows)
    # Inspect's own totals hold even without stored samples, so these stay exact.
    cover = coverage(log)
    # Everything below needs the samples; without them it is null, not a misleading zero.
    timed = bool(samples)
    return {
        "log": path,
        "model": log.eval.model,
        "task_args": log.eval.task_args,
        "config": {k: v for k, v in log.eval.config.model_dump().items() if v is not None},
        "warnings": warnings,
        "samples": cover["planned"],
        "errors": cover["errored"],
        "wall_clock_s": round(wall, 1) if wall is not None else None,
        # Output tokens of the model under test over the whole run: the throughput a
        # concurrency setting actually achieves.
        "run_out_tok_per_s": round(out_tokens / wall, 1) if timed and wall else None,
        "model_under_test": summarize(solver_rows) if timed else None,
        "by_group": {f"{k} / {t}": summarize(r) for (k, t), r in sorted(groups.items())} if timed else None,
        "judge": summarize(judge) if judge else None,
        "slowest_samples": [
            {"id": s.id, "epoch": s.epoch, "seconds": round(s.total_time, 1)}
            for s in sorted(samples, key=lambda s: s.total_time or 0, reverse=True)[:5]
            if s.total_time is not None
        ] if timed else None,
    }


def request_key(request, replies):
    """What a call sent, with the model's own earlier replies masked; plus any problems.

    ``request`` is the request body Inspect logged for the call (``ModelEvent.call``) and
    ``replies`` the response messages of the earlier turns of the same sample. The body is
    kept byte for byte except ``extra_headers``, which holds a random per-request id. Each
    assistant message must be the reply the model gave on that turn (Inspect resends the
    text with one leading newline added); it is then masked, since sampled replies differ
    between runs.
    """
    problems = []
    body = {k: v for k, v in request.items() if k != "extra_headers"}
    sent = [m for m in body["messages"] if m["role"] == "assistant"]
    if len(sent) != len(replies):
        problems.append(f"{len(sent)} replies in context, expected {len(replies)}")
    for message, reply in zip(sent, replies):
        expected = ("\n" + (reply.get("content") or ""), reply.get("reasoning"))
        if (message.get("content"), message.get("reasoning")) != expected:
            problems.append("a reply in context is not the model's own")
    body["messages"] = [{"role": "assistant"} if m["role"] == "assistant" else m for m in body["messages"]]
    return json.dumps(body, sort_keys=True), problems


def chat_call(event):
    """The request body and reply message of a logged call, if it is an OpenAI-style chat call.

    The prompt check reads Inspect's raw provider log (``ModelEvent.call``), which only has
    ``messages`` and ``choices`` for OpenAI-compatible chat endpoints such as Ollama's.
    """
    call = event.call
    try:
        call.request["messages"]
        return call.request, call.response["choices"][0]["message"]
    except (AttributeError, KeyError, IndexError, TypeError):
        raise ValueError(
            "--same-prompts needs the raw chat request and response in the log, as an "
            "OpenAI-compatible provider (e.g. ollama/...) records them; this log has a call without them"
        ) from None


def same_prompts(before_path, after_path):
    """Differences between two runs in what the model under test was sent (see module doc)."""
    problems = []

    def calls(path):
        out = {}
        for sample in read_eval_log(path, resolve_attachments=True).samples:
            events = model_calls(sample)
            chats = [chat_call(e) for e in events]
            for turn, (request, _) in enumerate(chats):
                key, found = request_key(request, [reply for _, reply in chats[:turn]])
                problems.extend(f"{path}: {sample.id} epoch {sample.epoch} turn {turn}: {p}" for p in found)
                out[(str(sample.id), sample.epoch, turn)] = key
        return out

    before, after = calls(before_path), calls(after_path)
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            problems.append(f"{key}: call only in {'after' if key in after else 'before'}")
        elif before[key] != after[key]:
            problems.append(f"{key}: request body differs")
    return len(set(before) & set(after)), problems


def judged(path):
    """Per (sample id, epoch): the score value and the judge prompts, from a scored log."""
    out = {}
    for sample in read_eval_log(path, resolve_attachments=True).samples:
        value = str(next(iter(sample.scores.values())).value) if sample.scores else None
        prompts = [m.text for e in model_calls(sample, "grader") for m in e.input]
        out[(str(sample.id), sample.epoch)] = (value, prompts)
    return out


def same_labels(before_path, after_path):
    """Label agreement and every difference between two scorings of the same transcripts."""
    return compare_judged(judged(before_path), judged(after_path))


def compare_judged(before, after):
    """``same_labels`` on two ``judged`` mappings: (turns, turns that agree, problems)."""
    problems, turns, agree = [], 0, 0
    for key in sorted(set(before) | set(after)):
        if key not in before or key not in after:
            problems.append(f"{key}: only in {'after' if key in after else 'before'}")
            continue
        (v1, p1), (v2, p2) = before[key], after[key]
        if p1 != p2:
            problems.append(f"{key}: judge prompts differ")
        l1, l2 = (v1 or "").split(">"), (v2 or "").split(">")
        turns += max(len(l1), len(l2))
        agree += sum(a == b for a, b in zip(l1, l2))
        if v1 != v2:
            problems.append(f"{key}: labels {v1} -> {v2}")
    return turns, agree, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--same-prompts", action="store_true", help="compare exactly two logs")
    parser.add_argument("--same-labels", action="store_true", help="compare two scored logs")
    args = parser.parse_args()
    if args.same_labels:
        if len(args.logs) != 2:
            parser.error("--same-labels takes exactly two logs")
        turns, agree, problems = same_labels(*args.logs)
        print("\n".join(problems))
        print(f"{'OK' if not problems else 'DIFFERENT'} - {agree}/{turns} turn labels agree")
        sys.exit(1 if problems else 0)
    if args.same_prompts:
        if len(args.logs) != 2:
            parser.error("--same-prompts takes exactly two logs")
        try:
            compared, problems = same_prompts(*args.logs)
        except ValueError as e:
            parser.exit(2, f"{e}\n")
        print("\n".join(problems) or f"OK - {compared} model calls send the same prompts and config")
        sys.exit(1 if problems else 0)
    reports = [report(p) for p in args.logs]
    for r in reports:
        for w in r["warnings"]:
            print(f"warning: {r['log']}: {w}", file=sys.stderr)
    print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()
