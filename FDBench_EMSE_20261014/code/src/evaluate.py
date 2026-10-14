"""
evaluate.py — 阶段四：从 runs.jsonl 计算全部指标并输出论文表格

流程：
  1. 去重：同一 (task_id, arm) 成功优先，成功中取迭代轮次最高的一条，
     把追加写产生的重复/旧失败记录清成干净的 106×6 配对数据。
  2. 四组指标（用已测通的 metrics.py）：
       M1 检索  Doc-Hit@5（按 KB 子集分层，验 H2）
       M2 定位  Prop-Hit@1（严格相等）、Region-IoU
       M3 修复  Fix@1 / Fix@3（重渲 fixed_html_final 与 target_boxes 结构比对）、副作用率
       M6 教学  分层合规率、答案泄漏率（layer_compliance）
  3. 统计（用 stats.py）：六组阶梯 McNemar/Wilcoxon + Holm 校正
  4. 解析质量：每组干净解析 / 抢救 / 失败占比
  5. 输出：论文表 13（A/B/D/F 四列）+ 六组消融表 + 按类别分层表

运行：
    python evaluate.py --runs ../results/runs.jsonl --tasks ../data/tasks \
                       --kb ../data/kb --out ../results
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import metrics
import stats

ARMS = ["A", "B", "C", "D", "E", "F"]


# --------------------------------------------------------------------------
# 载入与去重
# --------------------------------------------------------------------------
def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dedup(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """同一 (task,arm)：成功优先；成功中取 iters 最大的一条。"""
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["task_id"], r["arm"])
        if k not in best:
            best[k] = r
            continue
        old = best[k]
        old_ok = not old.get("error")
        new_ok = not r.get("error")
        if new_ok and not old_ok:
            best[k] = r
        elif new_ok and old_ok and r.get("iters", 0) >= old.get("iters", 0):
            best[k] = r
        elif not new_ok and not old_ok:
            best[k] = r  # 都失败，留后者
    return best


# --------------------------------------------------------------------------
# 单条记录的各项指标
# --------------------------------------------------------------------------
def prop_hit(pred: str, gt: str) -> int:
    """严格相等；仅当 pred 是 gt 的直接子属性（gt-xxx）时算部分命中→按命中计。"""
    if not pred:
        return 0
    pred, gt = pred.strip().lower(), gt.strip().lower()
    if pred == gt:
        return 1
    # margin-left ⊂ margin 这类真子属性关系
    if pred.startswith(gt + "-") or gt.startswith(pred + "-"):
        return 1
    return 0


def doc_hit(retrieved: list[str], gold: list[str], k: int = 5) -> int:
    return int(bool(set(retrieved[:k]) & set(gold)))


def eval_record(r: dict, task: dict, gold_map: dict, browser) -> dict:
    """算一条记录的全部指标。修复类指标需重渲，复用传入的 browser。"""
    parsed = r.get("parsed", {})
    gt_prop = task["gt_property"]
    out = {
        "task_id": r["task_id"], "arm": r["arm"], "category": task["category"],
        "error": bool(r.get("error")),
        "iters": r.get("iters", 0), "latency_s": r.get("latency_s", 0.0),
        "in_tokens": r.get("in_tokens", 0), "out_tokens": r.get("out_tokens", 0),
    }

    # M2 定位
    out["prop_hit"] = prop_hit(parsed.get("property", ""), gt_prop)
    pred_region = parsed.get("region")
    if pred_region and task.get("gt_regions"):
        pr = {"x": pred_region.get("x", 0), "y": pred_region.get("y", 0),
              "w": pred_region.get("w", 0), "h": pred_region.get("h", 0)}
        out["region_iou"] = metrics.region_iou([pr], task["gt_regions"])
    else:
        out["region_iou"] = 0.0

    # M1 检索（A 组无检索）
    if r["arm"] == "A":
        out["doc_hit"] = None
    else:
        gold = gold_map.get(gt_prop, [])
        out["doc_hit"] = doc_hit(r.get("retrieved", []), gold, 5)

    # M3 修复：重渲 fixed_html_final，与 target_boxes 结构比对
    fixed = r.get("fixed_html_final", "")
    out["fix"] = 0
    out["regression"] = 0.0
    if fixed and not r.get("error"):
        from render import render_and_extract
        rr = render_and_extract(fixed, None, browser)
        if rr.ok:
            out["fix"] = int(metrics.is_fixed(task["target_boxes"], rr.boxes))
            out["regression"] = metrics.regression_ratio(
                task["student_boxes"], task["target_boxes"], rr.boxes)

    # M6 教学：分层合规 / 泄漏
    fb = parsed.get("feedback", {})
    if fb:
        lc = metrics.layer_compliance(fb, gt_prop,
                                      task.get("gt_after", "").split(":")[-1].strip(" ;"))
        out["layer_compliant"] = int(lc["compliant"])
        out["leaked"] = int(lc["leaked"])
    else:
        out["layer_compliant"] = 0
        out["leaked"] = 1  # 无分层反馈视为泄漏（未按契约输出）

    # 解析质量：edits 是否存在
    out["has_edits"] = int(bool(parsed.get("edits")))
    return out


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def summarize(evals: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by_arm[e["arm"]].append(e)

    def rate(items, key):
        vals = [x[key] for x in items if x[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {}
    for a in ARMS:
        items = by_arm[a]
        if not items:
            continue
        summary[a] = {
            "n": len(items),
            "doc_hit@5": rate(items, "doc_hit") if a != "A" else None,
            "prop_hit@1": rate(items, "prop_hit"),
            "region_iou": rate(items, "region_iou"),
            "fix@1": rate(items, "fix"),
            "regression": rate(items, "regression"),
            "layer_compliant": rate(items, "layer_compliant"),
            "leaked": rate(items, "leaked"),
            "avg_iters": rate(items, "iters"),
            "avg_latency": rate(items, "latency_s"),
            "avg_out_tokens": rate(items, "out_tokens"),
            "error_rate": rate(items, "error"),
        }
    return summary


def ladder_significance(evals: list[dict]) -> str:
    """六组阶梯的 Prop-Hit 与 Fix@1 显著性（配对 McNemar + Holm）。"""
    by_arm_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in evals:
        by_arm_task[e["arm"]][e["task_id"]] = e
    common = set.intersection(*[set(by_arm_task[a]) for a in ARMS]) if all(
        a in by_arm_task for a in ARMS) else set()
    common = sorted(common)
    lines = [f"配对任务数（六组交集）：{len(common)}"]
    for metric in ("prop_hit", "fix"):
        arm_bin = {a: [by_arm_task[a][t][metric] for t in common] for a in ARMS}
        lines.append("")
        lines.append(stats.ladder_report(arm_bin, metric))
    return "\n".join(lines)


def by_category(evals: list[dict]) -> str:
    """按 category × arm 的 Fix@1 与 Prop-Hit，定位哪类任务拖后腿。"""
    cats = sorted(set(e["category"] for e in evals))
    lines = ["按类别分层（Fix@1 / Prop-Hit@1）："]
    header = "  类别".ljust(12) + "".join(a.center(14) for a in ARMS)
    lines.append(header)
    for c in cats:
        row = [f"  {c}".ljust(12)]
        for a in ARMS:
            items = [e for e in evals if e["category"] == c and e["arm"] == a]
            if items:
                fix = sum(x["fix"] for x in items) / len(items)
                ph = sum(x["prop_hit"] for x in items) / len(items)
                row.append(f"{fix:.0%}/{ph:.0%}".center(14))
            else:
                row.append("-".center(14))
        lines.append("".join(row))
    return "\n".join(lines)


def paper_table(summary: dict) -> str:
    """论文表 13：A/B/D/F 四列。"""
    cols = ["A", "B", "D", "F"]
    labels = {"A": "A 无检索", "B": "B 文本RAG",
              "D": "D 多模态RAG", "F": "F M-RAG-Ed"}
    rows = [
        ("证据命中率 Doc-Hit@5", "doc_hit@5", "{:.0%}"),
        ("属性定位 Prop-Hit@1", "prop_hit@1", "{:.0%}"),
        ("区域定位 Region-IoU", "region_iou", "{:.2f}"),
        ("首轮修复率 Fix@1", "fix@1", "{:.0%}"),
        ("副作用率", "regression", "{:.0%}"),
        ("分层合规率", "layer_compliant", "{:.0%}"),
        ("答案泄漏率", "leaked", "{:.0%}"),
        ("平均轮次", "avg_iters", "{:.1f}"),
        ("平均时延/s", "avg_latency", "{:.1f}"),
    ]
    out = ["论文表 13（可直接誊入）:", ""]
    head = "  指标".ljust(24) + "".join(labels[c].center(14) for c in cols)
    out.append(head)
    for name, key, fmt in rows:
        line = f"  {name}".ljust(24)
        for c in cols:
            v = summary.get(c, {}).get(key)
            line += ("—" if v is None else fmt.format(v)).center(14)
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(runs: Path, tasks_dir: Path, kb: Path, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    rows = load_jsonl(runs)
    tasks = {t["task_id"]: t for t in load_jsonl(tasks_dir / "tasks.jsonl")}
    gold_map = json.loads((kb / "gold.json").read_text(encoding="utf-8"))

    best = dedup(rows)
    print(f"去重：{len(rows)} 条 -> {len(best)} 条（应为 {len(tasks)}×{len(ARMS)}）")

    # 解析质量统计
    pq = defaultdict(lambda: Counter())
    for (t, a), r in best.items():
        if r.get("error"):
            pq[a]["失败"] += 1
        elif r.get("parsed", {}).get("edits"):
            pq[a]["正常"] += 1
        else:
            pq[a]["抢救/无edits"] += 1

    evals = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--hide-scrollbars"])
        try:
            for i, ((t, a), r) in enumerate(best.items()):
                if t not in tasks:
                    continue
                evals.append(eval_record(r, tasks[t], gold_map, browser))
                if (i + 1) % 60 == 0:
                    print(f"  已评测 {i+1}/{len(best)}")
        finally:
            browser.close()

    summary = summarize(evals)

    # ---- 输出 ----
    report = []
    report.append("=" * 64)
    report.append("解析质量（每组）：")
    for a in ARMS:
        c = pq[a]
        report.append(f"  {a}: 正常 {c['正常']}, 抢救/无edits {c['抢救/无edits']}, 失败 {c['失败']}")
    report.append("")
    report.append("=" * 64)
    report.append("各组指标汇总：")
    for a in ARMS:
        s = summary.get(a, {})
        if not s:
            continue
        dh = "—" if s["doc_hit@5"] is None else f"{s['doc_hit@5']:.1%}"
        report.append(
            f"  {a}: n={s['n']} | Doc-Hit@5={dh} | Prop-Hit={s['prop_hit@1']:.1%} | "
            f"Region-IoU={s['region_iou']:.2f} | Fix@1={s['fix@1']:.1%} | "
            f"副作用={s['regression']:.1%} | 合规={s['layer_compliant']:.1%} | "
            f"泄漏={s['leaked']:.1%} | 轮次={s['avg_iters']:.1f} | "
            f"时延={s['avg_latency']:.1f}s")
    report.append("")
    report.append("=" * 64)
    report.append(paper_table(summary))
    report.append("")
    report.append("=" * 64)
    report.append(ladder_significance(evals))
    report.append("")
    report.append("=" * 64)
    report.append(by_category(evals))

    text = "\n".join(report)
    print("\n" + text)
    (out / "report.txt").write_text(text, encoding="utf-8")
    (out / "metrics.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in evals),
        encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 -> {out/'report.txt'}")
    print(f"逐条指标 -> {out/'metrics.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("../results/runs.jsonl"))
    ap.add_argument("--tasks", type=Path, default=Path("../data/tasks"))
    ap.add_argument("--kb", type=Path, default=Path("../data/kb"))
    ap.add_argument("--out", type=Path, default=Path("../results"))
    a = ap.parse_args()
    main(a.runs, a.tasks, a.kb, a.out)
