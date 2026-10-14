#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FDBench Audit-v2 E6

Generation-stage evidence causality.

Conditions
----------
correct:
    Frozen D+ technical-query Top-5 retrieved evidence.

shuffled:
    One-to-one, leakage-free, no-document-overlap,
    length-matched irrelevant Top-5 evidence.

Design
------
- Same task
- Same original complaint
- Same target/student screenshots
- Same student HTML
- Same prompt template
- Same model configuration
- Only evidence content changes
- One generation round only
- Two independent replicates per condition
- Alternating execution order across tasks

No feedback / retry loop is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path.cwd()
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arms import (
    apply_edits,
    build_prompt,
    open_text,
    parse_json_loose,
)


EXP = (
    ROOT
    / "experiments_audit_v2"
    / "06_evidence_causality"
)

TASKS_DIR = ROOT / "data" / "tasks"

EVIDENCE_PATH = (
    EXP
    / "e6_clean_evidence.jsonl"
)

DEFAULT_OUT = (
    EXP
    / "e6_single_round_runs.jsonl"
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
                    str(
                        r["task_id"]
                    ),
                    str(
                        r["condition"]
                    ),
                    int(
                        r["replicate"]
                    ),
                )
            )

    return done


def hash_obj(obj):
    payload = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


def hash_text(text):
    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


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
                text.encode(
                    "utf-8"
                )
            ),
    }


def generate(
    task,
    condition,
    replicate,
    evidence,
    llm,
    execution_position,
    execution_order,
):
    t0 = time.time()

    student_html = open_text(
        task,
        "student",
    )

    error = ""

    try:
        # Use the historical F first-round prompt structure.
        # No feedback is present in E6.
        messages = build_prompt(
            task,
            "F",
            evidence,
            None,
        )

        prompt_sha = hash_obj(
            messages
        )

        prompt_json_chars = len(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

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
        prompt_sha = ""
        prompt_json_chars = 0
        raw = ""
        raw_sha = ""
        ti = 0
        to = 0
        parsed = {}
        edits = []
        fixed_html = ""
        error = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    lens = evidence_text_length(
        evidence
    )

    return {
        "task_id":
            str(
                task["task_id"]
            ),

        "experiment":
            "E6_single_round_evidence_causality",

        "condition":
            condition,

        "replicate":
            replicate,

        "execution_position":
            execution_position,

        "execution_order":
            execution_order,

        "one_round_only":
            True,

        "prompt_arm_template":
            "F_round1",

        "retrieved":
            [
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

        "prompt_json_chars":
            prompt_json_chars,

        "raw_response_sha256":
            raw_sha,

        "raw_response":
            raw,

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
            time.time()
            - t0,

        "error":
            error,
    }


def main(
    out_path,
    limit,
    resume,
):
    from llm_qwen import (
        llm,
    )

    tasks = load_tasks()

    if limit > 0:
        tasks = tasks[:limit]

    ev_rows = {
        str(r["task_id"]): r
        for r in load_jsonl(
            EVIDENCE_PATH
        )
    }

    if len(ev_rows) != 106:
        raise RuntimeError(
            f"expected 106 clean evidence rows, "
            f"got {len(ev_rows)}"
        )

    done = (
        load_done(
            out_path
        )
        if resume
        else set()
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "FDBench Audit-v2 E6"
    )
    print(
        "Single-round evidence causality"
    )
    print("=" * 96)

    print(
        "tasks =",
        len(tasks),
    )

    print(
        "replicates per condition = 2"
    )

    print(
        "conditions =",
        CONDITIONS,
    )

    print(
        "output =",
        out_path,
    )

    n_ok = 0
    n_fail = 0
    n_skip = 0

    with out_path.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for idx, task in enumerate(
            tasks,
            1,
        ):
            tid = str(
                task["task_id"]
            )

            erow = ev_rows[
                tid
            ]

            correct_ev = (
                erow[
                    "correct_evidence"
                ]
            )

            shuffled_ev = (
                erow[
                    "shuffled_evidence"
                ]
            )

            # Two alternating pairs.
            if idx % 2 == 1:
                schedule = [
                    ("correct", 1),
                    ("shuffled", 1),
                    ("correct", 2),
                    ("shuffled", 2),
                ]
            else:
                schedule = [
                    ("shuffled", 1),
                    ("correct", 1),
                    ("shuffled", 2),
                    ("correct", 2),
                ]

            order_labels = [
                f"{c}:r{r}"
                for c, r
                in schedule
            ]

            print()
            print(
                f"[{idx}/{len(tasks)}] "
                f"{tid} "
                f"order={order_labels}"
            )

            print(
                "  correct docs =",
                erow[
                    "correct_doc_ids"
                ],
            )

            print(
                "  shuffled docs =",
                erow[
                    "shuffled_doc_ids"
                ],
            )

            print(
                "  evidence char diff "
                "(correct-shuffled) =",
                erow[
                    "char_diff_correct_minus_shuffled"
                ],
            )

            for pos, (
                condition,
                replicate,
            ) in enumerate(
                schedule,
                1,
            ):
                key = (
                    tid,
                    condition,
                    replicate,
                )

                if key in done:
                    n_skip += 1

                    print(
                        f"  {condition} "
                        f"r{replicate}: "
                        "SKIP"
                    )

                    continue

                evidence = (
                    correct_ev
                    if condition
                    == "correct"
                    else shuffled_ev
                )

                rec = generate(
                    task=task,
                    condition=condition,
                    replicate=replicate,
                    evidence=evidence,
                    llm=llm,
                    execution_position=pos,
                    execution_order=order_labels,
                )

                # Frozen clean-shuffle provenance.
                rec[
                    "source_task_id"
                ] = (
                    tid
                    if condition
                    == "correct"
                    else erow[
                        "source_task_id"
                    ]
                )

                rec[
                    "target_gt_property"
                ] = erow[
                    "gt_property"
                ]

                rec[
                    "clean_shuffle_no_gold_leakage"
                ] = True

                rec[
                    "clean_shuffle_no_doc_overlap"
                ] = True

                fout.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

                if rec["error"]:
                    n_fail += 1

                    print(
                        f"  {condition} "
                        f"r{replicate}: "
                        f"FAIL "
                        f"{rec['error'][:160]}"
                    )

                else:
                    n_ok += 1

                    print(
                        f"  {condition} "
                        f"r{replicate}: "
                        f"in/out="
                        f"{rec['in_tokens']}/"
                        f"{rec['out_tokens']} "
                        f"latency="
                        f"{rec['latency_s']:.2f}s "
                        f"response="
                        f"{rec['raw_response_sha256'][:12]}"
                    )

    print()
    print("=" * 96)

    print(
        "done:",
        f"ok={n_ok}",
        f"fail={n_fail}",
        f"skip={n_skip}",
    )

    print(
        "output =",
        out_path,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
    )

    ap.set_defaults(
        resume=True
    )

    args = ap.parse_args()

    main(
        out_path=args.out,
        limit=args.limit,
        resume=args.resume,
    )
