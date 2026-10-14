#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path.cwd()

EXP = (
    ROOT
    / "experiments_audit_v2"
    / "10_human_validation"
)

PUBLIC = (
    EXP
    / "annotation_package_PUBLIC"
)

PRIVATE = (
    EXP
    / "PRIVATE_DO_NOT_SHARE"
)

ASSETS_A = PUBLIC / "A_construction" / "images"
ASSETS_B = PUBLIC / "B_metric_validity" / "images"
ASSETS_C = PUBLIC / "C_pair_preference" / "images"


TASKS_PATH = (
    ROOT
    / "data/tasks/tasks.jsonl"
)

E6_PATH = (
    ROOT
    / "experiments_audit_v2"
    / "06_evidence_causality"
    / "e6_single_round_runs.jsonl"
)

E8_PATH = (
    ROOT
    / "experiments_audit_v2"
    / "08_second_generator"
    / "e8_kimi_single_round_runs.jsonl"
)


SAMPLE_SEED = 20260907
PAIR_SIDE_SEED = 20260908
PRESENTATION_SEED = 20260909

N_SAMPLE_TASKS = 40

GENERATORS = (
    "qwen3.5-plus",
    "kimi-k2.5",
)

CONDITIONS = (
    "correct",
    "shuffled",
)


# ============================================================
# Generic helpers
# ============================================================

def load_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def write_json(path, obj):
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, rows):
    path.write_text(
        "\n".join(
            json.dumps(
                x,
                ensure_ascii=False,
                sort_keys=True,
            )
            for x in rows
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def resolve_project_path(raw):
    p = Path(
        str(raw)
    )

    candidates = []

    if p.is_absolute():
        candidates.append(
            p
        )

    else:
        candidates.extend([
            ROOT / p,
            ROOT / "data/tasks" / p,
            ROOT / "data" / p,
        ])

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        f"cannot resolve project path: {raw!r}; "
        f"tried {[str(x) for x in candidates]}"
    )


def copy_png(src, dst):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )

    if not dst.exists():
        raise RuntimeError(
            f"copy failed: {dst}"
        )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()

        for row in rows:
            w.writerow(
                {
                    k:
                        row.get(
                            k,
                            "",
                        )
                    for k
                    in fieldnames
                }
            )


def rel_public(path):
    return str(
        path.relative_to(
            PUBLIC
        )
    )


# ============================================================
# Clean output from any prior failed attempt
# ============================================================

if PUBLIC.exists():
    shutil.rmtree(
        PUBLIC
    )

if PRIVATE.exists():
    shutil.rmtree(
        PRIVATE
    )

PUBLIC.mkdir(
    parents=True,
    exist_ok=True,
)

PRIVATE.mkdir(
    parents=True,
    exist_ok=True,
)

ASSETS_A.mkdir(
    parents=True,
    exist_ok=True,
)

ASSETS_B.mkdir(
    parents=True,
    exist_ok=True,
)

ASSETS_C.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 108)
print(
    "E10.1b — Build frozen blinded human-validation package"
)
print("=" * 108)

print(
    "sample seed =",
    SAMPLE_SEED
)

print(
    "pair-side seed =",
    PAIR_SIDE_SEED
)

print(
    "presentation seed =",
    PRESENTATION_SEED
)


# ============================================================
# 1. Load and validate tasks
# ============================================================

tasks_rows = load_jsonl(
    TASKS_PATH
)

if len(tasks_rows) != 106:
    raise RuntimeError(
        f"expected 106 tasks, got {len(tasks_rows)}"
    )

tasks = {
    str(
        x["task_id"]
    ):
        x
    for x in tasks_rows
}

if len(tasks) != 106:
    raise RuntimeError(
        "task_id is not unique"
    )

task_ids = sorted(
    tasks.keys()
)


# ============================================================
# 2. Canonicalize E6
# ============================================================

e6_raw = load_jsonl(
    E6_PATH
)

e6_success = [
    x
    for x in e6_raw
    if not str(
        x.get(
            "error",
            "",
        )
    ).strip()
]

e6_fail = [
    x
    for x in e6_raw
    if str(
        x.get(
            "error",
            "",
        )
    ).strip()
]


e6_groups = defaultdict(
    list
)

for row in e6_success:

    key = (
        str(
            row[
                "task_id"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
        int(
            row[
                "replicate"
            ]
        ),
    )

    e6_groups[
        key
    ].append(
        row
    )


duplicate_success = {
    k: len(v)
    for k, v
    in e6_groups.items()
    if len(v) != 1
}

if duplicate_success:
    raise RuntimeError(
        "E6 has non-unique successful cells: "
        f"{list(duplicate_success.items())[:20]}"
    )


expected_e6_keys = {
    (
        tid,
        condition,
        replicate,
    )
    for tid in task_ids
    for condition in CONDITIONS
    for replicate in (
        1,
        2,
    )
}


if set(
    e6_groups.keys()
) != expected_e6_keys:

    missing = sorted(
        expected_e6_keys
        -
        set(
            e6_groups.keys()
        )
    )

    extra = sorted(
        set(
            e6_groups.keys()
        )
        -
        expected_e6_keys
    )

    raise RuntimeError(
        "E6 successful-cell coverage mismatch\n"
        f"missing={missing[:20]}\n"
        f"extra={extra[:20]}"
    )


e6 = {
    k:
        v[0]
    for k, v
    in e6_groups.items()
}


print()
print(
    "E6 raw / success / fail =",
    len(e6_raw),
    "/",
    len(e6_success),
    "/",
    len(e6_fail),
)

print(
    "E6 canonical successful cells =",
    len(e6),
)


# ============================================================
# 3. Canonicalize E8
# ============================================================

e8_raw = load_jsonl(
    E8_PATH
)

e8_success = [
    x
    for x in e8_raw
    if not str(
        x.get(
            "error",
            "",
        )
    ).strip()
]

if len(e8_success) != 212:
    raise RuntimeError(
        f"E8 successful rows != 212: {len(e8_success)}"
    )


e8_groups = defaultdict(
    list
)

for row in e8_success:

    key = (
        str(
            row[
                "task_id"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
    )

    e8_groups[
        key
    ].append(
        row
    )


bad_e8 = {
    k: len(v)
    for k, v
    in e8_groups.items()
    if len(v) != 1
}

if bad_e8:
    raise RuntimeError(
        f"E8 non-unique cells: {bad_e8}"
    )


expected_e8_keys = {
    (
        tid,
        condition,
    )
    for tid in task_ids
    for condition in CONDITIONS
}


if set(
    e8_groups.keys()
) != expected_e8_keys:

    raise RuntimeError(
        "E8 cell coverage mismatch"
    )


e8 = {
    k:
        v[0]
    for k, v
    in e8_groups.items()
}


for row in e8_success:

    if int(
        row.get(
            "replicate",
            -1,
        )
    ) != 1:
        raise RuntimeError(
            "E8 contains non-replicate-1 row"
        )

    if not bool(
        row.get(
            "prompt_matches_e6",
            False,
        )
    ):
        raise RuntimeError(
            "E8 prompt_matches_e6 != True"
        )

    if (
        str(
            row.get(
                "historical_e6_prompt_sha256",
                "",
            )
        )
        !=
        str(
            row.get(
                "prompt_sha256",
                "",
            )
        )
    ):
        raise RuntimeError(
            "E8 historical/current prompt SHA mismatch"
        )


# Directly confirm E8 matches E6 replicate 1.
prompt_pair_mismatches = []

for tid in task_ids:

    for condition in CONDITIONS:

        e6_row = e6[
            (
                tid,
                condition,
                1,
            )
        ]

        e8_row = e8[
            (
                tid,
                condition,
            )
        ]

        if (
            e6_row[
                "prompt_sha256"
            ]
            !=
            e8_row[
                "prompt_sha256"
            ]
        ):

            prompt_pair_mismatches.append(
                (
                    tid,
                    condition,
                )
            )


if prompt_pair_mismatches:
    raise RuntimeError(
        "E6-r1/E8 prompt mismatches: "
        f"{prompt_pair_mismatches[:20]}"
    )


print(
    "E8 canonical cells =",
    len(e8),
)

print(
    "E6-r1 vs E8 prompt SHA matches =",
    212,
    "/ 212",
)


# ============================================================
# 4. Validate candidate HTML
# ============================================================

empty_html = []

for tid in task_ids:

    for condition in CONDITIONS:

        q = e6[
            (
                tid,
                condition,
                1,
            )
        ]

        k = e8[
            (
                tid,
                condition,
            )
        ]

        for label, row in (
            (
                "qwen",
                q,
            ),
            (
                "kimi",
                k,
            ),
        ):

            if not str(
                row.get(
                    "fixed_html_final",
                    "",
                )
            ).strip():

                empty_html.append(
                    (
                        tid,
                        label,
                        condition,
                    )
                )


if empty_html:
    raise RuntimeError(
        f"empty candidate HTML: {empty_html}"
    )


# ============================================================
# 5. Freeze 40-task random sample
# ============================================================

sample_rng = random.Random(
    SAMPLE_SEED
)

sampled_tasks = sorted(
    sample_rng.sample(
        task_ids,
        N_SAMPLE_TASKS,
    )
)


sample_manifest_private = {
    "sampling_unit":
        "task",

    "population_n":
        106,

    "sample_n":
        N_SAMPLE_TASKS,

    "seed":
        SAMPLE_SEED,

    "sampled_task_ids":
        sampled_tasks,
}


write_json(
    PRIVATE
    / "e10_sample_40_tasks.json",
    sample_manifest_private,
)


print()
print(
    "sampled tasks =",
    len(sampled_tasks),
)

print(
    "sample IDs =",
    sampled_tasks,
)


# ============================================================
# 6. Import project render / metric implementation
# ============================================================

sys.path.insert(
    0,
    str(
        ROOT
        / "src"
    ),
)

from playwright.sync_api import sync_playwright
from render import render_and_extract
from metrics import is_fixed


# ============================================================
# 7. Render all 160 candidate outputs once
#    and independently compute automatic Fix.
#
#    automatic_fix is PRIVATE ONLY.
# ============================================================

render_root = (
    PRIVATE
    / "canonical_candidate_renders"
)

render_root.mkdir(
    parents=True,
    exist_ok=True,
)


candidate_records = []

render_failures = []


def get_candidate(
    tid,
    generator,
    condition,
):

    if generator == "qwen3.5-plus":

        row = e6[
            (
                tid,
                condition,
                1,
            )
        ]

        source = (
            "E6",
            1,
        )

    elif generator == "kimi-k2.5":

        row = e8[
            (
                tid,
                condition,
            )
        ]

        source = (
            "E8",
            1,
        )

    else:
        raise ValueError(
            generator
        )

    return (
        row,
        source,
    )


print()
print(
    "Rendering 160 frozen repair candidates..."
)


with sync_playwright() as pw:

    browser = pw.chromium.launch(
        args=[
            "--hide-scrollbars"
        ]
    )

    try:

        n = 0

        for tid in sampled_tasks:

            task = tasks[
                tid
            ]

            for generator in GENERATORS:

                for condition in CONDITIONS:

                    n += 1

                    row, source = get_candidate(
                        tid,
                        generator,
                        condition,
                    )

                    safe_generator = (
                        "qwen"
                        if generator
                        ==
                        "qwen3.5-plus"
                        else
                        "kimi"
                    )

                    canonical_png = (
                        render_root
                        /
                        (
                            f"{tid}"
                            f"__{safe_generator}"
                            f"__{condition}.png"
                        )
                    )

                    rr = render_and_extract(
                        str(
                            row[
                                "fixed_html_final"
                            ]
                        ),
                        canonical_png,
                        browser,
                    )

                    if not rr.ok:

                        render_failures.append({
                            "task_id":
                                tid,

                            "generator":
                                generator,

                            "condition":
                                condition,

                            "error":
                                str(
                                    rr.error
                                ),
                        })

                        continue

                    if not canonical_png.exists():

                        raise RuntimeError(
                            "render_and_extract succeeded "
                            "but PNG was not created: "
                            f"{canonical_png}"
                        )

                    automatic_fix = bool(
                        is_fixed(
                            task[
                                "target_boxes"
                            ],
                            rr.boxes,
                        )
                    )

                    candidate_records.append({
                        "task_id":
                            tid,

                        "generator":
                            generator,

                        "condition":
                            condition,

                        "source_experiment":
                            source[0],

                        "replicate":
                            source[1],

                        "prompt_sha256":
                            row[
                                "prompt_sha256"
                            ],

                        "raw_response_sha256":
                            row.get(
                                "raw_response_sha256",
                                "",
                            ),

                        "fixed_html_sha256":
                            hashlib.sha256(
                                str(
                                    row[
                                        "fixed_html_final"
                                    ]
                                ).encode(
                                    "utf-8"
                                )
                            ).hexdigest(),

                        "canonical_render":
                            str(
                                canonical_png.relative_to(
                                    ROOT
                                )
                            ),

                        "canonical_render_sha256":
                            sha256_file(
                                canonical_png
                            ),

                        "automatic_fix":
                            automatic_fix,
                    })

                    if (
                        n % 20 == 0
                        or
                        n == 160
                    ):
                        print(
                            f"  rendered {n}/160"
                        )

    finally:
        browser.close()


if render_failures:
    raise RuntimeError(
        "candidate rerender failures:\n"
        +
        json.dumps(
            render_failures,
            ensure_ascii=False,
            indent=2,
        )
    )


if len(candidate_records) != 160:
    raise RuntimeError(
        "expected 160 candidate records, "
        f"got {len(candidate_records)}"
    )


candidate_key = {
    (
        x[
            "task_id"
        ],
        x[
            "generator"
        ],
        x[
            "condition"
        ],
    ):
        x
    for x in candidate_records
}


if len(candidate_key) != 160:
    raise RuntimeError(
        "candidate canonical key not unique"
    )


print(
    "candidate rerender = 160/160 PASS"
)


# ============================================================
# 8. E10-A construction-validity package
#
#    Hide all task IDs.
# ============================================================

presentation_rng = random.Random(
    PRESENTATION_SEED
)

a_task_order = list(
    task_ids
)

presentation_rng.shuffle(
    a_task_order
)


a_private = []

a_public = []

for idx, tid in enumerate(
    a_task_order,
    start=1,
):

    case_id = (
        f"A-{idx:03d}"
    )

    task = tasks[
        tid
    ]

    target_src = resolve_project_path(
        task[
            "target_png"
        ]
    )

    student_src = resolve_project_path(
        task[
            "student_png"
        ]
    )

    target_dst = (
        ASSETS_A
        /
        f"{case_id}_target.png"
    )

    student_dst = (
        ASSETS_A
        /
        f"{case_id}_student.png"
    )

    copy_png(
        target_src,
        target_dst,
    )

    copy_png(
        student_src,
        student_dst,
    )

    a_public.append({
        "case_id":
            case_id,

        "target_image":
            rel_public(
                target_dst
            ),

        "student_image":
            rel_public(
                student_dst
            ),
    })

    a_private.append({
        "case_id":
            case_id,

        "task_id":
            tid,

        "category":
            task.get(
                "category"
            ),

        "gt_property":
            task.get(
                "gt_property"
            ),

        "op_id":
            task.get(
                "op_id"
            ),
    })


write_jsonl(
    PUBLIC
    / "A_construction"
    / "manifest.jsonl",
    a_public,
)

write_json(
    PRIVATE
    / "e10_A_blind_key.json",
    a_private,
)


# ============================================================
# 9. E10-B candidate package
#
#    160 candidates are globally shuffled and each candidate
#    receives an anonymous ID unrelated to task/model/condition.
# ============================================================

b_hidden = []

for tid in sampled_tasks:

    for generator in GENERATORS:

        for condition in CONDITIONS:

            b_hidden.append({
                "task_id":
                    tid,

                "generator":
                    generator,

                "condition":
                    condition,
            })


presentation_rng.shuffle(
    b_hidden
)


b_public = []

b_private = []


for idx, hidden in enumerate(
    b_hidden,
    start=1,
):

    candidate_id = (
        f"B-{idx:03d}"
    )

    tid = hidden[
        "task_id"
    ]

    generator = hidden[
        "generator"
    ]

    condition = hidden[
        "condition"
    ]

    task = tasks[
        tid
    ]

    c = candidate_key[
        (
            tid,
            generator,
            condition,
        )
    ]

    target_src = resolve_project_path(
        task[
            "target_png"
        ]
    )

    student_src = resolve_project_path(
        task[
            "student_png"
        ]
    )

    repaired_src = (
        ROOT
        /
        c[
            "canonical_render"
        ]
    )

    target_dst = (
        ASSETS_B
        /
        f"{candidate_id}_target.png"
    )

    student_dst = (
        ASSETS_B
        /
        f"{candidate_id}_student.png"
    )

    repaired_dst = (
        ASSETS_B
        /
        f"{candidate_id}_repaired.png"
    )

    copy_png(
        target_src,
        target_dst,
    )

    copy_png(
        student_src,
        student_dst,
    )

    copy_png(
        repaired_src,
        repaired_dst,
    )

    b_public.append({
        "candidate_id":
            candidate_id,

        "target_image":
            rel_public(
                target_dst
            ),

        "student_image":
            rel_public(
                student_dst
            ),

        "repaired_image":
            rel_public(
                repaired_dst
            ),
    })

    b_private.append({
        "candidate_id":
            candidate_id,

        "task_id":
            tid,

        "generator":
            generator,

        "condition":
            condition,

        "replicate":
            1,

        "source_experiment":
            c[
                "source_experiment"
            ],

        "automatic_fix":
            c[
                "automatic_fix"
            ],

        "prompt_sha256":
            c[
                "prompt_sha256"
            ],

        "raw_response_sha256":
            c[
                "raw_response_sha256"
            ],

        "fixed_html_sha256":
            c[
                "fixed_html_sha256"
            ],

        "render_sha256":
            c[
                "canonical_render_sha256"
            ],
    })


write_jsonl(
    PUBLIC
    / "B_metric_validity"
    / "manifest.jsonl",
    b_public,
)

write_json(
    PRIVATE
    / "e10_B_blind_key.json",
    b_private,
)


# ============================================================
# 10. E10-C paired package
#
#     Exact 20/20 A/B balance WITHIN EACH generator.
# ============================================================

pair_rng = random.Random(
    PAIR_SIDE_SEED
)

c_hidden = []


for generator in GENERATORS:

    task_order = list(
        sampled_tasks
    )

    pair_rng.shuffle(
        task_order
    )

    # Exactly 20 correct=A and 20 correct=B
    correct_is_a_flags = (
        [True] * 20
        +
        [False] * 20
    )

    pair_rng.shuffle(
        correct_is_a_flags
    )

    for tid, correct_is_a in zip(
        task_order,
        correct_is_a_flags,
    ):

        c_hidden.append({
            "task_id":
                tid,

            "generator":
                generator,

            "correct_is_a":
                correct_is_a,
        })


# Presentation order independent from pair-side assignment.
presentation_rng.shuffle(
    c_hidden
)


c_public = []

c_private = []


for idx, hidden in enumerate(
    c_hidden,
    start=1,
):

    pair_id = (
        f"C-{idx:03d}"
    )

    tid = hidden[
        "task_id"
    ]

    generator = hidden[
        "generator"
    ]

    correct_is_a = bool(
        hidden[
            "correct_is_a"
        ]
    )

    condition_a = (
        "correct"
        if correct_is_a
        else "shuffled"
    )

    condition_b = (
        "shuffled"
        if correct_is_a
        else "correct"
    )

    task = tasks[
        tid
    ]

    ca = candidate_key[
        (
            tid,
            generator,
            condition_a,
        )
    ]

    cb = candidate_key[
        (
            tid,
            generator,
            condition_b,
        )
    ]

    target_src = resolve_project_path(
        task[
            "target_png"
        ]
    )

    student_src = resolve_project_path(
        task[
            "student_png"
        ]
    )

    a_src = (
        ROOT
        /
        ca[
            "canonical_render"
        ]
    )

    b_src = (
        ROOT
        /
        cb[
            "canonical_render"
        ]
    )

    target_dst = (
        ASSETS_C
        /
        f"{pair_id}_target.png"
    )

    student_dst = (
        ASSETS_C
        /
        f"{pair_id}_student.png"
    )

    a_dst = (
        ASSETS_C
        /
        f"{pair_id}_A.png"
    )

    b_dst = (
        ASSETS_C
        /
        f"{pair_id}_B.png"
    )

    copy_png(
        target_src,
        target_dst,
    )

    copy_png(
        student_src,
        student_dst,
    )

    copy_png(
        a_src,
        a_dst,
    )

    copy_png(
        b_src,
        b_dst,
    )

    c_public.append({
        "pair_id":
            pair_id,

        "target_image":
            rel_public(
                target_dst
            ),

        "student_image":
            rel_public(
                student_dst
            ),

        "candidate_A_image":
            rel_public(
                a_dst
            ),

        "candidate_B_image":
            rel_public(
                b_dst
            ),
    })

    c_private.append({
        "pair_id":
            pair_id,

        "task_id":
            tid,

        "generator":
            generator,

        "A_condition":
            condition_a,

        "B_condition":
            condition_b,

        "A_automatic_fix":
            ca[
                "automatic_fix"
            ],

        "B_automatic_fix":
            cb[
                "automatic_fix"
            ],

        "A_render_sha256":
            ca[
                "canonical_render_sha256"
            ],

        "B_render_sha256":
            cb[
                "canonical_render_sha256"
            ],
    })


write_jsonl(
    PUBLIC
    / "C_pair_preference"
    / "manifest.jsonl",
    c_public,
)

write_json(
    PRIVATE
    / "e10_C_blind_key.json",
    c_private,
)


# ============================================================
# 11. Exact balance validation
# ============================================================

balance = {}

for generator in GENERATORS:

    rows = [
        x
        for x
        in c_private
        if x[
            "generator"
        ]
        ==
        generator
    ]

    counts = Counter(
        x[
            "A_condition"
        ]
        for x
        in rows
    )

    balance[
        generator
    ] = dict(
        counts
    )

    if (
        counts[
            "correct"
        ]
        !=
        20
        or
        counts[
            "shuffled"
        ]
        !=
        20
    ):
        raise RuntimeError(
            f"A/B balance failed for "
            f"{generator}: {counts}"
        )


# ============================================================
# 12. Annotation templates
# ============================================================

a_template_rows = [
    {
        "case_id":
            x[
                "case_id"
            ],

        "defect_visible":
            "",

        "single_primary_defect":
            "",

        "page_intact":
            "",

        "css_like_defect":
            "",

        "notes":
            "",
    }
    for x in a_public
]


b_template_rows = [
    {
        "candidate_id":
            x[
                "candidate_id"
            ],

        "defect_resolution":
            "",

        "new_regression":
            "",

        "target_similarity_1_5":
            "",

        "notes":
            "",
    }
    for x in b_public
]


c_template_rows = [
    {
        "pair_id":
            x[
                "pair_id"
            ],

        "pair_preference":
            "",

        "notes":
            "",
    }
    for x in c_public
]


write_csv(
    PUBLIC
    / "A_construction"
    / "annotation_template.csv",

    [
        "case_id",
        "defect_visible",
        "single_primary_defect",
        "page_intact",
        "css_like_defect",
        "notes",
    ],

    a_template_rows,
)


write_csv(
    PUBLIC
    / "B_metric_validity"
    / "annotation_template.csv",

    [
        "candidate_id",
        "defect_resolution",
        "new_regression",
        "target_similarity_1_5",
        "notes",
    ],

    b_template_rows,
)


write_csv(
    PUBLIC
    / "C_pair_preference"
    / "annotation_template.csv",

    [
        "pair_id",
        "pair_preference",
        "notes",
    ],

    c_template_rows,
)


# ============================================================
# 13. Public rater instructions
# ============================================================

README = r"""# E10 人工验证独立评分说明

## 一、评分原则

请独立完成评分，不与其他评分者讨论，也不要查看其他评分者的结果。

只依据所给图片进行判断。不要尝试推断图片来自哪个模型、哪种实验条件或哪个原始任务。

如确实无法可靠判断，请使用规定的 `U`，不要强行选择。

---

# A：基准构造有效性

每个案例包含：

- `target_image`：目标页面
- `student_image`：待修复页面

请回答：

### defect_visible

学生页面与目标页面之间是否存在清晰可见的视觉缺陷？

允许值：

- `Yes`
- `No`
- `U`

### single_primary_defect

差异是否主要表现为一个占主导地位的缺陷，而不是多个彼此独立的问题？

允许值：

- `Yes`
- `No`
- `U`

### page_intact

学生页面是否仍是结构完整、可正常理解和调试的页面，而不是严重损坏或渲染失败？

允许值：

- `Yes`
- `No`
- `U`

### css_like_defect

从视觉表现看，该差异是否像一个可通过局部 HTML/CSS 的布局、尺寸、间距、字体或样式修改解决的问题？

允许值：

- `Yes`
- `No`
- `U`

---

# B：自动修复判定有效性

每个案例包含：

- `target_image`
- `student_image`
- `repaired_image`

## defect_resolution

修复后的页面对原主要缺陷的解决程度：

- `0`：未解决，主要缺陷仍明显存在
- `1`：部分解决，已有改善但仍存在明显相关差异
- `2`：已解决，主要缺陷已经消失，达到目标或可接受的等价效果
- `U`：无法可靠判断

## new_regression

修复是否引入原 student 页面没有的明显新视觉/结构问题：

- `0`：没有明显新增问题
- `1`：存在明显新增问题
- `U`：无法判断

## target_similarity_1_5

修复页与目标页整体接近程度：

- `1`：明显不接近
- `2`：较不接近
- `3`：部分接近
- `4`：较接近
- `5`：基本一致

若确实无法判断，请在 notes 中说明，不要随意填写。

---

# C：两种修复结果成对比较

每个案例包含：

- `target_image`
- `student_image`
- `candidate_A_image`
- `candidate_B_image`

综合考虑：

1. 原始主要缺陷是否被解决；
2. 与目标页面的接近程度；
3. 是否产生新的视觉或结构问题。

填写 `pair_preference`：

- `-2`：A 明显优于 B
- `-1`：A 略优于 B
- `0`：两者总体相当
- `+1`：B 略优于 A
- `+2`：B 明显优于 A

---

# 注意事项

不要：

- 根据文件编号猜测条件；
- 根据个人对某种模型的印象评分；
- 修改案例顺序；
- 删除案例；
- 与其他评分者协商后再填；
- 在全部评分完成之前索取实验条件对应关系。

评分完成后仅提交三个 CSV 文件即可。
"""


(
    PUBLIC
    / "README_人工标注说明.md"
).write_text(
    README,
    encoding="utf-8",
)


# ============================================================
# 14. Frozen protocol
# ============================================================

protocol = {
    "experiment":
        "FDBench Audit-v2 E10 human validation",

    "stage":
        "E10.1b protocol and blind-package freeze",

    "status":
        "PRE_ANNOTATION_FROZEN",

    "raters":
        3,

    "independent_rating":
        True,

    "sampling": {
        "population_tasks":
            106,

        "sample_tasks_B_C":
            40,

        "sampling_unit":
            "task",

        "sampling_seed":
            SAMPLE_SEED,

        "sampling_method":
            (
                "simple random sample without replacement "
                "from all 106 task IDs"
            ),
    },

    "blinding": {
        "task_id_hidden":
            True,

        "generator_hidden":
            True,

        "condition_hidden":
            True,

        "automatic_fix_hidden":
            True,

        "evidence_hidden":
            True,

        "ground_truth_fields_hidden":
            True,

        "pair_side_seed":
            PAIR_SIDE_SEED,

        "presentation_seed":
            PRESENTATION_SEED,

        "pair_side_balance":
            (
                "exactly 20 correct-as-A and "
                "20 correct-as-B within each generator"
            ),
    },

    "E10_A": {
        "unit":
            "task",

        "n":
            106,

        "questions": {
            "defect_visible":
                [
                    "Yes",
                    "No",
                    "U",
                ],

            "single_primary_defect":
                [
                    "Yes",
                    "No",
                    "U",
                ],

            "page_intact":
                [
                    "Yes",
                    "No",
                    "U",
                ],

            "css_like_defect":
                [
                    "Yes",
                    "No",
                    "U",
                ],
        },

        "primary_construction_valid_definition":
            (
                "majority Yes on defect_visible AND "
                "single_primary_defect AND page_intact"
            ),
    },

    "E10_B": {
        "unit":
            "repaired candidate",

        "tasks":
            40,

        "generators": [
            "Qwen3.5-Plus",
            "Kimi-K2.5",
        ],

        "conditions": [
            "correct",
            "clean shuffled",
        ],

        "qwen_replicate":
            1,

        "n_candidates":
            160,

        "questions": {
            "defect_resolution":
                [
                    "0",
                    "1",
                    "2",
                    "U",
                ],

            "new_regression":
                [
                    "0",
                    "1",
                    "U",
                ],

            "target_similarity_1_5":
                [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                ],
        },

        "human_fix_definition":
            (
                "defect_resolution == 2 AND "
                "new_regression == 0"
            ),

        "unresolved_definition":
            (
                "at least 2 of 3 raters give U "
                "on a required human-fix component"
            ),
    },

    "E10_C": {
        "unit":
            "(task, generator) paired repair",

        "tasks":
            40,

        "generators":
            2,

        "n_pairs":
            80,

        "question":
            "pair_preference",

        "scale": {
            "-2":
                "A clearly better",

            "-1":
                "A slightly better",

            "0":
                "tie",

            "+1":
                "B slightly better",

            "+2":
                "B clearly better",
        },
    },

    "agreement_metrics": {
        "E10_A":
            [
                "raw agreement",
                "Fleiss kappa",
                "Gwet AC1",
            ],

        "E10_B_defect_resolution":
            [
                "Krippendorff alpha ordinal",
            ],

        "E10_B_new_regression":
            [
                "raw agreement",
                "Fleiss kappa",
                "Gwet AC1",
            ],

        "E10_B_similarity":
            [
                "ICC(2,1) absolute agreement",
                "ICC(2,3) mean-rating reliability",
            ],

        "E10_C":
            [
                "Krippendorff alpha ordinal",
                "raw agreement on collapsed A/tie/B",
                "Gwet AC1 on collapsed A/tie/B",
            ],
    },

    "E10_C_effect_analysis": {
        "per_generator":
            True,

        "outcomes":
            [
                "correct better",
                "tie",
                "shuffled better",
                "net preference",
            ],

        "inference":
            [
                "task bootstrap 95% CI",
                "exact sign test excluding ties",
            ],
    },
}


write_json(
    PRIVATE
    / "e10_protocol_frozen.json",
    protocol,
)


# ============================================================
# 15. Source hashes
# ============================================================

source_hashes = {
    str(
        TASKS_PATH.relative_to(
            ROOT
        )
    ):
        sha256_file(
            TASKS_PATH
        ),

    str(
        E6_PATH.relative_to(
            ROOT
        )
    ):
        sha256_file(
            E6_PATH
        ),

    str(
        E8_PATH.relative_to(
            ROOT
        )
    ):
        sha256_file(
            E8_PATH
        ),

    "src/render.py":
        sha256_file(
            ROOT
            / "src/render.py"
        ),

    "src/metrics.py":
        sha256_file(
            ROOT
            / "src/metrics.py"
        ),
}


write_json(
    PRIVATE
    / "e10_source_sha256.json",
    source_hashes,
)


# ============================================================
# 16. Blind-key master
# ============================================================

master = {
    "WARNING":
        (
            "PRIVATE. DO NOT SHARE WITH RATERS "
            "UNTIL ALL RATINGS ARE LOCKED."
        ),

    "sample":
        sample_manifest_private,

    "A":
        a_private,

    "B":
        b_private,

    "C":
        c_private,

    "pair_side_balance":
        balance,

    "source_hashes":
        source_hashes,
}


write_json(
    PRIVATE
    / "e10_blind_key_MASTER.json",
    master,
)


# ============================================================
# 17. Create identical blank templates for 3 raters
# ============================================================

for rater in (
    "R1",
    "R2",
    "R3",
):

    rdir = (
        PUBLIC
        / "rater_templates"
        / rater
    )

    rdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        PUBLIC
        / "A_construction"
        / "annotation_template.csv",

        rdir
        / f"{rater}_A.csv",
    )

    shutil.copy2(
        PUBLIC
        / "B_metric_validity"
        / "annotation_template.csv",

        rdir
        / f"{rater}_B.csv",
    )

    shutil.copy2(
        PUBLIC
        / "C_pair_preference"
        / "annotation_template.csv",

        rdir
        / f"{rater}_C.csv",
    )


# ============================================================
# 18. Public and private hashes
# ============================================================

def hash_tree(
    root,
    exclude_names=None,
):
    exclude_names = set(
        exclude_names
        or []
    )

    rows = []

    for path in sorted(
        x
        for x in root.rglob(
            "*"
        )
        if x.is_file()
    ):

        if path.name in exclude_names:
            continue

        rows.append({
            "path":
                str(
                    path.relative_to(
                        root
                    )
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                path.stat().st_size,
        })

    return rows


public_hashes = hash_tree(
    PUBLIC
)

write_json(
    PUBLIC
    / "SHA256_manifest.json",
    public_hashes,
)


private_hashes = hash_tree(
    PRIVATE,
    exclude_names={
        "SHA256_manifest.json",
    },
)

write_json(
    PRIVATE
    / "SHA256_manifest.json",
    private_hashes,
)


# ============================================================
# 19. Final structural validation
# ============================================================

checks = {
    "tasks_population":
        len(
            tasks
        )
        ==
        106,

    "e6_successful_cells":
        len(
            e6
        )
        ==
        424,

    "e8_successful_cells":
        len(
            e8
        )
        ==
        212,

    "e6_e8_prompt_match":
        len(
            prompt_pair_mismatches
        )
        ==
        0,

    "sample_40_unique":
        (
            len(
                sampled_tasks
            )
            ==
            40
            and
            len(
                set(
                    sampled_tasks
                )
            )
            ==
            40
        ),

    "candidate_renders":
        len(
            candidate_records
        )
        ==
        160,

    "A_public_cases":
        len(
            a_public
        )
        ==
        106,

    "B_public_candidates":
        len(
            b_public
        )
        ==
        160,

    "C_public_pairs":
        len(
            c_public
        )
        ==
        80,

    "qwen_pair_balance":
        (
            balance[
                "qwen3.5-plus"
            ].get(
                "correct"
            )
            ==
            20
            and
            balance[
                "qwen3.5-plus"
            ].get(
                "shuffled"
            )
            ==
            20
        ),

    "kimi_pair_balance":
        (
            balance[
                "kimi-k2.5"
            ].get(
                "correct"
            )
            ==
            20
            and
            balance[
                "kimi-k2.5"
            ].get(
                "shuffled"
            )
            ==
            20
        ),
}


package_pass = all(
    checks.values()
)


summary = {
    "experiment":
        "E10.1b",

    "status":
        (
            "FROZEN"
            if package_pass
            else "FAILED"
        ),

    "seeds": {
        "sample":
            SAMPLE_SEED,

        "pair_side":
            PAIR_SIDE_SEED,

        "presentation":
            PRESENTATION_SEED,
    },

    "E6": {
        "physical_rows":
            len(
                e6_raw
            ),

        "successful_rows":
            len(
                e6_success
            ),

        "failed_rows":
            len(
                e6_fail
            ),

        "canonical_cells":
            len(
                e6
            ),
    },

    "E8": {
        "physical_rows":
            len(
                e8_raw
            ),

        "canonical_cells":
            len(
                e8
            ),
    },

    "sample_n":
        len(
            sampled_tasks
        ),

    "candidate_renders":
        len(
            candidate_records
        ),

    "A_cases":
        len(
            a_public
        ),

    "B_candidates":
        len(
            b_public
        ),

    "C_pairs":
        len(
            c_public
        ),

    "pair_side_balance":
        balance,

    "checks":
        checks,

    "package_pass":
        package_pass,

    "public_package":
        str(
            PUBLIC
        ),

    "private_package":
        str(
            PRIVATE
        ),
}


write_json(
    EXP
    / "e10_1b_summary.json",
    summary,
)


# ============================================================
# 20. Report
# ============================================================

lines = [
    "=" * 108,
    "E10.1b blinded human-validation package report",
    "=" * 108,
    "",
    f"sample seed = {SAMPLE_SEED}",
    f"pair-side seed = {PAIR_SIDE_SEED}",
    f"presentation seed = {PRESENTATION_SEED}",
    "",
    "INPUT CANONICALIZATION",
    "-" * 108,
    (
        f"E6 physical / success / fail = "
        f"{len(e6_raw)} / "
        f"{len(e6_success)} / "
        f"{len(e6_fail)}"
    ),
    (
        f"E6 canonical successful cells = "
        f"{len(e6)}"
    ),
    (
        f"E8 canonical cells = "
        f"{len(e8)}"
    ),
    (
        "E6 replicate-1 vs E8 prompt SHA matches = "
        "212 / 212"
    ),
    "",
    "SAMPLE",
    "-" * 108,
    (
        f"population = 106 tasks"
    ),
    (
        f"sample = {len(sampled_tasks)} tasks"
    ),
    (
        "sampling = simple random sample "
        "without replacement"
    ),
    "",
    "BLINDED PACKAGE",
    "-" * 108,
    (
        f"E10-A cases = "
        f"{len(a_public)}"
    ),
    (
        f"E10-B candidates = "
        f"{len(b_public)}"
    ),
    (
        f"E10-C pairs = "
        f"{len(c_public)}"
    ),
    (
        f"candidate rerender = "
        f"{len(candidate_records)}/160 PASS"
    ),
    "",
    "PAIR-SIDE BALANCE",
    "-" * 108,
    (
        f"Qwen A-side condition counts = "
        f"{balance['qwen3.5-plus']}"
    ),
    (
        f"Kimi A-side condition counts = "
        f"{balance['kimi-k2.5']}"
    ),
    "",
    "CHECKS",
    "-" * 108,
]


for k, v in checks.items():
    lines.append(
        f"{k} = {v}"
    )


lines += [
    "",
    "=" * 108,
    (
        f"BLIND_PACKAGE_PASS = "
        f"{package_pass}"
    ),
    "=" * 108,
    "",
    (
        "PUBLIC package:"
    ),
    str(
        PUBLIC
    ),
    "",
    (
        "PRIVATE package — DO NOT SHARE WITH RATERS:"
    ),
    str(
        PRIVATE
    ),
]


report_path = (
    EXP
    / "e10_1b_report.txt"
)

report_path.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)


print()
print(
    "\n".join(
        lines
    )
)


if not package_pass:
    raise RuntimeError(
        "E10.1b blind-package validation failed"
    )
