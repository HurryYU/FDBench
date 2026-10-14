#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path.cwd()
SRC = ROOT / "src"

E8DIR = (
    ROOT
    / "experiments_audit_v2"
    / "08_second_generator"
)

E6DIR = (
    ROOT
    / "experiments_audit_v2"
    / "06_evidence_causality"
)

TASKS_DIR = ROOT / "data" / "tasks"

EVIDENCE_PATH = (
    E6DIR
    / "e6_clean_evidence.jsonl"
)

E6_RUNS = (
    E6DIR
    / "e6_single_round_runs.jsonl"
)

DEFAULT_OUT = (
    E8DIR
    / "e8_kimi_single_round_runs.jsonl"
)

PREFLIGHT_JSON = (
    E8DIR
    / "e8_4_preflight.json"
)

PREFLIGHT_TXT = (
    E8DIR
    / "e8_4_preflight.txt"
)


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if str(E8DIR) not in sys.path:
    sys.path.insert(0, str(E8DIR))


from arms import (
    apply_edits,
    build_prompt,
    open_text,
    parse_json_loose,
)

from llm_kimi import (
    BASE_URL,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    llm,
)


CONDITIONS = (
    "correct",
    "shuffled",
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def hash_obj(obj):
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def hash_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_tasks():
    tasks = load_jsonl(
        TASKS_DIR
        / "tasks.jsonl"
    )

    root = str(
        TASKS_DIR.resolve()
    )

    for task in tasks:
        task["_root"] = root

        for key in (
            "target_png",
            "student_png",
        ):
            task[key] = str(
                (
                    TASKS_DIR
                    / task[key]
                ).resolve()
            )

    return tasks


def evidence_text_length(ev):
    text = "\n".join(
        (
            f"[{i}] "
            f"{x.get('title', '')}\n"
            f"{x.get('snippet', '')}"
        )
        for i, x
        in enumerate(ev, 1)
    )

    return {
        "chars":
            len(text),

        "bytes":
            len(
                text.encode("utf-8")
            ),
    }


def load_done(path):
    done = set()

    if not path.exists():
        return done

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            r = json.loads(line)
        except Exception:
            continue

        if not r.get("error"):
            done.add(
                (
                    str(r["task_id"]),
                    str(r["condition"]),
                )
            )

    return done


def load_historical_prompt_hashes():
    rows = load_jsonl(
        E6_RUNS
    )

    by_key = {}

    for r in rows:
        if r.get("error"):
            continue

        key = (
            str(r["task_id"]),
            str(r["condition"]),
        )

        sha = str(
            r.get(
                "prompt_sha256",
                ""
            )
        )

        if not sha:
            continue

        by_key.setdefault(
            key,
            set(),
        ).add(sha)

    expected = {
        (f"T{i:04d}", c)
        for i in range(106)
        for c in CONDITIONS
    }

    if set(by_key) != expected:
        missing = sorted(
            expected
            - set(by_key)
        )

        raise RuntimeError(
            "historical E6 prompt keys incomplete: "
            f"got={len(by_key)}, "
            f"expected={len(expected)}, "
            f"missing={missing[:10]}"
        )

    bad = {
        k: sorted(v)
        for k, v in by_key.items()
        if len(v) != 1
    }

    if bad:
        raise RuntimeError(
            "E6 successful replicates do not share "
            "a unique prompt SHA for some cells: "
            + repr(
                list(bad.items())[:5]
            )
        )

    return {
        k: next(iter(v))
        for k, v in by_key.items()
    }


def schedule_for_task(index):
    # Balance condition order over the 106 tasks.
    if index % 2 == 1:
        return (
            "correct",
            "shuffled",
        )

    return (
        "shuffled",
        "correct",
    )


def run_preflight():
    tasks = load_tasks()

    evidence_rows = {
        str(r["task_id"]): r
        for r in load_jsonl(
            EVIDENCE_PATH
        )
    }

    hist_sha = (
        load_historical_prompt_hashes()
    )

    if len(tasks) != 106:
        raise RuntimeError(
            f"expected 106 tasks, got {len(tasks)}"
        )

    if len(evidence_rows) != 106:
        raise RuntimeError(
            "expected 106 evidence rows, "
            f"got {len(evidence_rows)}"
        )

    planned = []
    prompt_match = 0
    prompt_distinct = 0
    overlap_violations = []
    evidence_size_violations = []

    correct_first = 0
    shuffled_first = 0

    for index, task in enumerate(
        tasks,
        1,
    ):
        tid = str(
            task["task_id"]
        )

        erow = evidence_rows[
            tid
        ]

        correct_ev = erow[
            "correct_evidence"
        ]

        shuffled_ev = erow[
            "shuffled_evidence"
        ]

        if (
            len(correct_ev) != 5
            or
            len(shuffled_ev) != 5
        ):
            evidence_size_violations.append(
                tid
            )

        cdocs = {
            str(x.get("doc_id", ""))
            for x in correct_ev
        }

        sdocs = {
            str(x.get("doc_id", ""))
            for x in shuffled_ev
        }

        overlap = sorted(
            cdocs
            &
            sdocs
        )

        if overlap:
            overlap_violations.append({
                "task_id": tid,
                "overlap": overlap,
            })

        schedule = schedule_for_task(
            index
        )

        if schedule[0] == "correct":
            correct_first += 1
        else:
            shuffled_first += 1

        prompt_shas = {}

        for position, condition in enumerate(
            schedule,
            1,
        ):
            evidence = (
                correct_ev
                if condition == "correct"
                else shuffled_ev
            )

            messages = build_prompt(
                task,
                "F",
                evidence,
                None,
            )

            current_sha = hash_obj(
                messages
            )

            historical_sha = hist_sha[
                (
                    tid,
                    condition,
                )
            ]

            match = (
                current_sha
                ==
                historical_sha
            )

            prompt_match += int(
                match
            )

            prompt_shas[
                condition
            ] = current_sha

            planned.append({
                "task_id":
                    tid,

                "task_index":
                    index,

                "condition":
                    condition,

                "execution_position":
                    position,

                "prompt_sha256":
                    current_sha,

                "historical_e6_prompt_sha256":
                    historical_sha,

                "prompt_matches_e6":
                    match,

                "doc_ids": [
                    str(
                        x.get(
                            "doc_id",
                            ""
                        )
                    )
                    for x in evidence
                ],
            })

        prompt_distinct += int(
            prompt_shas["correct"]
            !=
            prompt_shas["shuffled"]
        )

    checks = {
        "tasks":
            len(tasks),

        "planned_cells":
            len(planned),

        "conditions_per_task":
            2,

        "correct_first_tasks":
            correct_first,

        "shuffled_first_tasks":
            shuffled_first,

        "prompt_matches_historical_e6":
            prompt_match,

        "correct_shuffled_prompt_distinct_tasks":
            prompt_distinct,

        "evidence_size_violations":
            len(
                evidence_size_violations
            ),

        "correct_shuffled_doc_overlap_violations":
            len(
                overlap_violations
            ),
    }

    passed = (
        checks["tasks"] == 106
        and
        checks["planned_cells"] == 212
        and
        checks["correct_first_tasks"] == 53
        and
        checks["shuffled_first_tasks"] == 53
        and
        checks[
            "prompt_matches_historical_e6"
        ] == 212
        and
        checks[
            "correct_shuffled_prompt_distinct_tasks"
        ] == 106
        and
        checks[
            "evidence_size_violations"
        ] == 0
        and
        checks[
            "correct_shuffled_doc_overlap_violations"
        ] == 0
    )

    manifest = {
        "experiment":
            "E8_second_generator_validation",

        "generator":
            MODEL,

        "endpoint":
            BASE_URL,

        "temperature":
            TEMPERATURE,

        "max_tokens":
            MAX_TOKENS,

        "enable_thinking":
            False,

        "response_format":
            "json_object",

        "prompt_template":
            "F_round1",

        "conditions":
            list(CONDITIONS),

        "planned_replicates_per_condition":
            1,

        "order_rule":
            (
                "odd task index: correct->shuffled; "
                "even task index: shuffled->correct"
            ),

        "checks":
            checks,

        "preflight_pass":
            passed,

        "input_hashes": {
            "tasks_jsonl":
                sha256_file(
                    TASKS_DIR
                    / "tasks.jsonl"
                ),

            "e6_clean_evidence":
                sha256_file(
                    EVIDENCE_PATH
                ),

            "e6_single_round_runs":
                sha256_file(
                    E6_RUNS
                ),

            "arms_py":
                sha256_file(
                    SRC
                    / "arms.py"
                ),

            "llm_kimi_py":
                sha256_file(
                    E8DIR
                    / "llm_kimi.py"
                ),
        },

        "planned_cells":
            planned,

        "violations": {
            "evidence_size":
                evidence_size_violations,

            "doc_overlap":
                overlap_violations,
        },
    }

    PREFLIGHT_JSON.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "=" * 96,
        "FDBench Audit-v2 E8.4 — Formal-run preflight",
        "=" * 96,
        "",
        f"model = {MODEL}",
        f"endpoint = {BASE_URL}",
        f"temperature = {TEMPERATURE}",
        f"max_tokens = {MAX_TOKENS}",
        "enable_thinking = False",
        "response_format = json_object",
        "prompt template = F_round1",
        "",
        "DESIGN",
        "-" * 96,
        "tasks = 106",
        "conditions = correct / shuffled",
        "replicates per condition = 1",
        "planned calls = 212",
        (
            "order = odd task index: correct->shuffled; "
            "even task index: shuffled->correct"
        ),
        "",
        "VALIDATION",
        "-" * 96,
    ]

    for k, v in checks.items():
        lines.append(
            f"{k} = {v}"
        )

    lines += [
        "",
        "=" * 96,
        f"PREFLIGHT_PASS = {passed}",
        "=" * 96,
        "",
        f"Output: {PREFLIGHT_JSON}",
        f"Output: {PREFLIGHT_TXT}",
    ]

    PREFLIGHT_TXT.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n".join(lines)
    )

    if not passed:
        raise RuntimeError(
            "E8 preflight failed; formal calls are blocked."
        )

    return (
        tasks,
        evidence_rows,
        hist_sha,
    )


def generate(
    task,
    condition,
    evidence,
    historical_prompt_sha,
    task_index,
    execution_position,
    execution_order,
):
    t0 = time.time()

    student_html = open_text(
        task,
        "student",
    )

    messages = build_prompt(
        task,
        "F",
        evidence,
        None,
    )

    prompt_sha = hash_obj(
        messages
    )

    if (
        prompt_sha
        !=
        historical_prompt_sha
    ):
        raise RuntimeError(
            f"prompt SHA mismatch for "
            f"{task['task_id']} / {condition}"
        )

    prompt_json_chars = len(
        json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    error = ""
    raw = ""
    raw_sha = ""
    parsed = {}
    edits = []
    fixed_html = ""
    ti = 0
    to = 0

    try:
        raw, ti, to = llm(
            messages
        )

        parsed = parse_json_loose(
            raw
        )

        edits = parsed.get(
            "edits",
            [],
        )

        if edits:
            fixed_html = apply_edits(
                student_html,
                edits,
            )
        else:
            fixed_html = parsed.get(
                "fixed_html",
                student_html,
            )

        raw_sha = hash_text(
            raw
        )

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    lens = evidence_text_length(
        evidence
    )

    return {
        "task_id":
            str(task["task_id"]),

        "experiment":
            "E8_second_generator_single_round",

        "generator":
            MODEL,

        "condition":
            condition,

        "replicate":
            1,

        "task_index":
            task_index,

        "execution_position":
            execution_position,

        "execution_order":
            execution_order,

        "one_round_only":
            True,

        "prompt_arm_template":
            "F_round1",

        "retrieved": [
            str(
                x.get(
                    "doc_id",
                    ""
                )
            )
            for x in evidence
        ],

        "evidence":
            evidence,

        "evidence_chars":
            lens["chars"],

        "evidence_bytes":
            lens["bytes"],

        "prompt_sha256":
            prompt_sha,

        "historical_e6_prompt_sha256":
            historical_prompt_sha,

        "prompt_matches_e6":
            True,

        "prompt_json_chars":
            prompt_json_chars,

        "raw_response":
            raw,

        "raw_response_sha256":
            raw_sha,

        "parsed":
            parsed,

        "edits":
            edits,

        "fixed_html_final":
            fixed_html,

        "in_tokens":
            ti,

        "out_tokens":
            to,

        "latency_s":
            time.time() - t0,

        "error":
            error,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    ap.add_argument(
        "--resume",
        action="store_true",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    tasks, evidence_rows, hist_sha = (
        run_preflight()
    )

    if args.dry_run:
        print()
        print(
            "DRY_RUN: no model calls were made."
        )
        return

    if not os.environ.get(
        "DASHSCOPE_API_KEY"
    ):
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未设置；正式运行已阻止。"
        )

    done = (
        load_done(args.out)
        if args.resume
        else set()
    )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 96)
    print(
        "FDBench Audit-v2 E8"
    )
    print(
        "Second-generator validation — Kimi K2.5"
    )
    print("=" * 96)
    print(
        "planned cells = 212"
    )
    print(
        "output =",
        args.out
    )
    print(
        "resume =",
        args.resume
    )

    n_ok = 0
    n_fail = 0
    n_skip = 0

    with args.out.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for index, task in enumerate(
            tasks,
            1,
        ):
            tid = str(
                task["task_id"]
            )

            erow = evidence_rows[
                tid
            ]

            schedule = schedule_for_task(
                index
            )

            order_label = "->".join(
                schedule
            )

            print()
            print(
                f"[{index}/106] "
                f"{tid} "
                f"order={order_label}"
            )

            for position, condition in enumerate(
                schedule,
                1,
            ):
                key = (
                    tid,
                    condition,
                )

                if key in done:
                    n_skip += 1
                    print(
                        f"  {condition}: SKIP"
                    )
                    continue

                evidence = (
                    erow[
                        "correct_evidence"
                    ]
                    if condition == "correct"
                    else
                    erow[
                        "shuffled_evidence"
                    ]
                )

                row = generate(
                    task=task,
                    condition=condition,
                    evidence=evidence,
                    historical_prompt_sha=
                        hist_sha[key],
                    task_index=index,
                    execution_position=position,
                    execution_order=order_label,
                )

                fout.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

                if row["error"]:
                    n_fail += 1

                    print(
                        f"  {condition}: "
                        f"ERROR {row['error']}"
                    )

                else:
                    n_ok += 1

                    print(
                        f"  {condition}: "
                        f"OK "
                        f"in/out="
                        f"{row['in_tokens']}/"
                        f"{row['out_tokens']} "
                        f"latency="
                        f"{row['latency_s']:.2f}s"
                    )

    print()
    print("=" * 96)
    print("RUN SUMMARY")
    print("=" * 96)
    print("success =", n_ok)
    print("failed  =", n_fail)
    print("skipped =", n_skip)
    print("output  =", args.out)


if __name__ == "__main__":
    main()
