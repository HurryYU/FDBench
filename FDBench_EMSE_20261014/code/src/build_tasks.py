"""
build_tasks.py — 阶段一：从 Design2Code 自动构造前端调试任务集（完整版·高产出率）

相对初版的三处关键改动：
  1. 每份源码遍历所有可用算子（而非只随机试一个），一份源码可贡献多个任务，
     源码利用率成倍提升——这是产出率偏低的根本修复。
  2. 目标页只渲染一次并缓存，其包围盒树在该源码的所有算子间复用，渲染开销减半。
  3. MIN_OFFSET_PX 由 8 降到 3：真实网页上大量有效变异只造成 3–6px 位移，
     8px 会把它们误杀。3px 配合“至少 1 元素错位”仍足以保证肉眼可辨。

运行：
    python build_tasks.py --src ./data/design2code/testset_final \
                          --out ./data/tasks --n 120 --seed 20260723
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from mutate import MUTATORS, CATEGORIES, apply_mutation, applicable_ops, make_complaint
from metrics import compare_trees, ground_truth_regions

# --- 有效性过滤阈值 -------------------------------------------------------
MIN_MISMATCH = 1          # 至少 1 个元素错位，否则变异不可见 -> 丢弃
MAX_MISMATCH_RATIO = 0.6  # 错位元素占比过高说明页面整体崩塌 -> 丢弃
MIN_OFFSET_PX = 3.0       # 最大位移低于此值判为不可觉察（初版为 8.0，真实页面偏严）
MIN_TOKENS, MAX_TOKENS = 400, 8000
MAX_PER_PAGE = 3          # 同一源码最多贡献几个任务，避免任务集被少数页面主导


def rough_tokens(text: str) -> int:
    return len(text) // 3


def load_pool(src_dir: Path) -> list[tuple[str, str]]:
    pool = []
    for f in sorted(src_dir.glob("*.html")):
        html = f.read_text(encoding="utf-8", errors="ignore")
        n = rough_tokens(html)
        if MIN_TOKENS <= n <= MAX_TOKENS:
            pool.append((f.stem, html))
    return pool


def build(src_dir: Path, out_dir: Path, n_target: int, seed: int) -> None:
    from playwright.sync_api import sync_playwright
    from render import render_and_extract

    rng = random.Random(seed)
    pool = load_pool(src_dir)
    rng.shuffle(pool)
    print(f"候选页面 {len(pool)} 份（已按 {MIN_TOKENS}-{MAX_TOKENS} token 过滤）")

    per_cat = n_target // len(CATEGORIES)
    quota = {c: per_cat for c in CATEGORIES}
    made = Counter()
    tasks, rejected = [], Counter()
    pages_used = 0

    (out_dir / "html").mkdir(parents=True, exist_ok=True)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--hide-scrollbars"])
        try:
            for page_id, html in pool:
                if all(made[c] >= quota[c] for c in CATEGORIES):
                    break

                # 该源码上所有可用算子；随机打乱以免类别偏置
                avail = applicable_ops(html, rng)
                rng.shuffle(avail)
                if not avail:
                    continue

                # 目标页渲染一次并缓存（所有算子共用）
                r_ok = render_and_extract(
                    html, out_dir / "img" / f"{page_id}_target.png", browser)
                if not r_ok.ok or len(r_ok.boxes) < 5:
                    rejected["原始页渲染失败或元素过少"] += 1
                    continue

                made_from_this = 0
                for op_id in avail:
                    if made_from_this >= MAX_PER_PAGE:
                        break
                    cat = _category_of(op_id)
                    if made[cat] >= quota[cat]:      # 该类已满，跳过
                        continue

                    mut = apply_mutation(html, op_id, rng)
                    if mut is None:
                        rejected["算子无目标"] += 1
                        continue

                    stu_png = out_dir / "img" / f"{page_id}_{op_id}_student.png"
                    r_bad = render_and_extract(mut.mutated_html, stu_png, browser)
                    if not r_bad.ok:
                        rejected["变异页渲染失败"] += 1
                        continue

                    cmp_ = compare_trees(r_ok.boxes, r_bad.boxes)
                    n_mis = len(cmp_["mismatches"])
                    ratio = n_mis / max(len(r_ok.boxes), 1)
                    max_off = max((m["offset_px"] for m in cmp_["mismatches"]),
                                  default=0.0)

                    if n_mis < MIN_MISMATCH:
                        rejected["变异不可见"] += 1
                        stu_png.unlink(missing_ok=True)
                        continue
                    if ratio > MAX_MISMATCH_RATIO:
                        rejected["页面整体崩塌"] += 1
                        stu_png.unlink(missing_ok=True)
                        continue
                    if max_off < MIN_OFFSET_PX and cmp_["missing"] == 0:
                        rejected["位移过小"] += 1
                        stu_png.unlink(missing_ok=True)
                        continue

                    tid = f"T{len(tasks):04d}"
                    (out_dir / "html" / f"{tid}_target.html").write_text(
                        html, encoding="utf-8")
                    (out_dir / "html" / f"{tid}_student.html").write_text(
                        mut.mutated_html, encoding="utf-8")

                    task = {
                        "task_id": tid,
                        "source_page": page_id,
                        "category": cat,
                        "complaint": make_complaint(mut.symptom_hint, rng),
                        **mut.to_dict(),
                        "target_boxes": r_ok.boxes,
                        "student_boxes": r_bad.boxes,
                        "gt_regions": ground_truth_regions(r_ok.boxes, r_bad.boxes),
                        "diff_stats": {"n_mismatch": n_mis,
                                       "ratio": round(ratio, 4),
                                       "max_offset_px": round(max_off, 2)},
                        "target_png": f"img/{page_id}_target.png",
                        "student_png": f"img/{page_id}_{op_id}_student.png",
                    }
                    tasks.append(task)
                    made[cat] += 1
                    made_from_this += 1
                    if len(tasks) % 10 == 0:
                        print(f"  已生成 {len(tasks)} 个任务 {dict(made)}")

                if made_from_this:
                    pages_used += 1
        finally:
            browser.close()

    (out_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks),
        encoding="utf-8")

    print(f"\n完成：{len(tasks)} 个任务，来自 {pages_used} 份源码 "
          f"-> {out_dir/'tasks.jsonl'}")
    print("类别分布:", dict(made))
    print("丢弃原因:", dict(rejected))
    print("\n请按手册 3.6 节对 10% 样本做人工抽检后再进入下一阶段。")


_CAT_CACHE: dict[str, str] = {}


def _category_of(op_id: str) -> str:
    if op_id not in _CAT_CACHE:
        probe_html = ("<style>.a{display:flex;justify-content:center;align-items:center;"
                      "box-sizing:border-box;padding:1px;width:50%;margin:0 auto;"
                      "max-width:9px;text-align:center;font-size:16px;font-weight:bold;"
                      "letter-spacing:1px}"
                      ".b{position:relative;z-index:1;line-height:1.5;height:9px;"
                      "flex-direction:row}</style>")
        r = apply_mutation(probe_html, op_id, random.Random(0))
        _CAT_CACHE[op_id] = r.category if r else "layout"
    return _CAT_CACHE[op_id]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./data/tasks"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260723)
    a = ap.parse_args()
    build(a.src, a.out, a.n, a.seed)
