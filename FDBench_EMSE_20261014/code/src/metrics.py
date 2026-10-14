"""
metrics.py — 全部自动评测指标的实现

指标分五组（与手册第 6 章一一对应）：
  M1 证据检索：Doc-Hit@k / Doc-nDCG@5
  M2 缺陷定位：Prop-Hit@1 / Line-Hit@1 / Region-IoU / Region-Acc@0.5
  M3 修复成效：Fix@1 / Fix@k / 副作用率 Regression
  M4 渲染一致性：StructSim / 元素匹配率 / 中心偏移
  M6 教学质量：分层合规率 / 答案泄漏率

M5（效率）由运行日志直接统计，不在此模块。
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

# ==========================================================================
# 基础：包围盒运算
# ==========================================================================

Box = dict  # {"path","tag","x","y","w","h","text"}


def iou(a: Box, b: Box) -> float:
    """两个包围盒的交并比。"""
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def center_offset(a: Box, b: Box) -> float:
    """两个包围盒中心点的欧氏距离（像素）。"""
    acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    return math.hypot(acx - bcx, acy - bcy)


def size_rel_diff(a: Box, b: Box) -> float:
    """尺寸相对差异，取宽高相对差的较大者。"""
    dw = abs(a["w"] - b["w"]) / max(a["w"], b["w"], 1.0)
    dh = abs(a["h"] - b["h"]) / max(a["h"], b["h"], 1.0)
    return max(dw, dh)


def index_by_path(boxes: Iterable[Box]) -> dict[str, Box]:
    return {b["path"]: b for b in boxes}


# ==========================================================================
# M4 渲染一致性 + M3 修复判定
# ==========================================================================

# 判定阈值。CENTER_TOL 取 3px 是为了容忍亚像素舍入而不放过真实位移；
# 若源码含随机字体回退，可放宽到 5px 并在论文中声明。
CENTER_TOL = 3.0
SIZE_TOL = 0.02


def compare_trees(target: list[Box], actual: list[Box],
                  center_tol: float = CENTER_TOL,
                  size_tol: float = SIZE_TOL) -> dict:
    """
    结构比对：把目标渲染与实际渲染的包围盒树逐元素对齐。

    返回：
      matched      两边都存在的元素数
      aligned      位置与尺寸均在容差内的元素数
      missing      目标有、实际无
      extra        实际有、目标无
      struct_sim   aligned / |target|，即结构相似度（主指标）
      mean_offset  匹配元素的平均中心偏移
      mismatches   未对齐元素的明细（用于生成教学反馈）
    """
    ti, ai = index_by_path(target), index_by_path(actual)
    common = set(ti) & set(ai)
    aligned, offsets, mismatches = 0, [], []

    for p in sorted(common):
        t, a = ti[p], ai[p]
        off, sd = center_offset(t, a), size_rel_diff(t, a)
        offsets.append(off)
        if off <= center_tol and sd <= size_tol:
            aligned += 1
        else:
            mismatches.append({
                "path": p, "tag": t["tag"],
                "offset_px": round(off, 2), "size_diff": round(sd, 4),
                "dx": round(a["x"] - t["x"], 2), "dy": round(a["y"] - t["y"], 2),
                "target_box": {k: t[k] for k in "xywh"},
                "actual_box": {k: a[k] for k in "xywh"},
            })

    n_target = max(len(ti), 1)
    return {
        "matched": len(common),
        "aligned": aligned,
        "missing": len(set(ti) - set(ai)),
        "extra": len(set(ai) - set(ti)),
        "struct_sim": aligned / n_target,
        "mean_offset": sum(offsets) / len(offsets) if offsets else 0.0,
        "mismatches": sorted(mismatches, key=lambda m: -m["offset_px"]),
    }


def is_fixed(target: list[Box], actual: list[Box],
             strict: bool = True, **kw) -> bool:
    """
    修复成功判据。

    strict=True（主判据）：结构完全还原 —— 无缺失、无多余、全部元素对齐。
    strict=False（宽松判据）：struct_sim >= 0.98 且无缺失元素，
        用于敏感性分析，报告两套结果可显著提升结论稳健性。
    """
    c = compare_trees(target, actual, **kw)
    if strict:
        return (c["missing"] == 0 and c["extra"] == 0
                and c["aligned"] == c["matched"] == len(target))
    return c["missing"] == 0 and c["struct_sim"] >= 0.98


def regression_ratio(baseline_buggy: list[Box], target: list[Box],
                     actual: list[Box], **kw) -> float:
    """
    副作用率：修复过程中"本来就对、改完反而错了"的元素占比。
    这是纯 Fix@1 掩盖不了的重要维度——大模型常整段重写导致连带破坏。
    """
    was_ok = {m["path"] for m in target} - {
        m["path"] for m in compare_trees(target, baseline_buggy, **kw)["mismatches"]}
    now_bad = {m["path"] for m in compare_trees(target, actual, **kw)["mismatches"]}
    if not was_ok:
        return 0.0
    return len(was_ok & now_bad) / len(was_ok)


def ground_truth_regions(target: list[Box], buggy: list[Box], **kw) -> list[Box]:
    """
    自动生成"视觉异常区域"真值：变异前后包围盒发生显著变化的元素。
    这是 Region-IoU 的黄金标准，全程无需人工标注。
    """
    c = compare_trees(target, buggy, **kw)
    bi = index_by_path(buggy)
    return [bi[m["path"]] for m in c["mismatches"] if m["path"] in bi]


# ==========================================================================
# M2 缺陷定位
# ==========================================================================

def prop_hit(pred_property: Optional[str], gt_property: str) -> int:
    """属性级定位命中。大小写与前后空白不敏感。"""
    if not pred_property:
        return 0
    return int(pred_property.strip().lower() == gt_property.strip().lower())


def line_hit(pred_line: Optional[int], gt_line: int, tol: int = 0) -> int:
    """行级定位命中。tol=0 为严格，tol=1 允许相邻行（应对删除型变异的歧义）。"""
    if pred_line is None:
        return 0
    return int(abs(int(pred_line) - gt_line) <= tol)


def region_iou(pred_boxes: list[Box], gt_boxes: list[Box]) -> float:
    """预测区域与真值区域的最佳匹配 IoU（取各真值框的最大 IoU 后平均）。"""
    if not gt_boxes:
        return 0.0
    if not pred_boxes:
        return 0.0
    scores = [max(iou(g, p) for p in pred_boxes) for g in gt_boxes]
    return sum(scores) / len(scores)


# ==========================================================================
# M1 证据检索
# ==========================================================================

def hit_at_k(ranked_doc_ids: list[str], gold_ids: set[str], k: int) -> int:
    return int(bool(set(ranked_doc_ids[:k]) & gold_ids))


def ndcg_at_k(ranked_doc_ids: list[str], gold_ids: set[str], k: int = 5) -> float:
    dcg = sum(1.0 / math.log2(i + 2)
              for i, d in enumerate(ranked_doc_ids[:k]) if d in gold_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ==========================================================================
# M6 教学质量（自动部分）
# ==========================================================================

_CODE_BLOCK = re.compile(r"```|<style|{\s*[a-z-]+\s*:", re.IGNORECASE)


def layer_compliance(layers: dict[str, str], gt_property: str,
                     gt_value: str = "") -> dict:
    """
    分层反馈合规性自动检查。layers 形如
        {"L1": "...现象...", "L2": "...概念...", "L3": "...证据...", "L4": "...实现..."}

    规则（对应手册 5.4 节）：
      R1 L1 不得出现 CSS 属性名          —— 只描述现象
      R2 L2 可含概念，不得含具体取值      —— 点到概念为止
      R3 L3 必须含证据来源标识            —— 可溯源
      R4 L4 之前不得出现代码块            —— 不提前给答案
    返回各规则通过与否，以及总的"答案泄漏"标志。
    """
    L1, L2 = layers.get("L1", ""), layers.get("L2", "")
    L3, L4 = layers.get("L3", ""), layers.get("L4", "")
    prop = gt_property.lower()

    r1 = prop not in L1.lower()
    r2 = (not gt_value) or (gt_value.lower() not in L2.lower())
    r3 = bool(re.search(r"(mdn|来源|出处|参见|文档|§|第\s*\d+\s*页|\[\d+\])",
                        L3, re.IGNORECASE))
    r4 = not bool(_CODE_BLOCK.search(L1 + L2 + L3))

    leaked = (not r1) or (not r4)
    return {"R1_no_prop_in_L1": r1, "R2_no_value_in_L2": r2,
            "R3_evidence_cited": r3, "R4_no_code_before_L4": r4,
            "compliant": all([r1, r2, r3, r4]), "leaked": leaked}


# ==========================================================================
# 自测
# ==========================================================================

if __name__ == "__main__":
    def B(path, x, y, w=100, h=50, tag="div"):
        return {"path": path, "tag": tag, "x": x, "y": y, "w": w, "h": h, "text": ""}

    target = [B("div[1]", 100, 0), B("div[1]/p[1]", 110, 10), B("div[2]", 100, 60)]
    buggy = [B("div[1]", 0, 0), B("div[1]/p[1]", 10, 10), B("div[2]", 100, 60)]
    fixed = [B("div[1]", 100, 0), B("div[1]/p[1]", 110, 10), B("div[2]", 100, 60)]
    partial = [B("div[1]", 100, 0), B("div[1]/p[1]", 110, 10), B("div[2]", 40, 60)]

    c = compare_trees(target, buggy)
    assert c["matched"] == 3 and c["aligned"] == 1, c
    assert abs(c["struct_sim"] - 1 / 3) < 1e-9
    assert c["mismatches"][0]["dx"] == -100.0

    assert is_fixed(target, fixed) is True
    assert is_fixed(target, buggy) is False
    assert is_fixed(target, partial) is False

    gt_regions = ground_truth_regions(target, buggy)
    assert len(gt_regions) == 2, gt_regions
    assert region_iou([B("div[1]", 0, 0)], [gt_regions[0]]) > 0.9
    assert region_iou([B("zzz", 900, 900)], gt_regions) == 0.0

    # 副作用：修好了 div[1] 却把原本正确的 div[2] 弄坏
    bad_fix = [B("div[1]", 100, 0), B("div[1]/p[1]", 110, 10), B("div[2]", 300, 60)]
    rr = regression_ratio(buggy, target, bad_fix)
    assert rr > 0, rr

    assert prop_hit("Justify-Content ", "justify-content") == 1
    assert prop_hit("display", "justify-content") == 0
    assert line_hit(12, 12) == 1 and line_hit(13, 12) == 0 and line_hit(13, 12, tol=1) == 1

    docs = ["css/display", "css/justify-content", "css/flex"]
    assert hit_at_k(docs, {"css/justify-content"}, 1) == 0
    assert hit_at_k(docs, {"css/justify-content"}, 5) == 1
    assert 0.6 < ndcg_at_k(docs, {"css/justify-content"}) < 0.7

    good = {"L1": "这一栏贴在了容器左侧，右侧留出大片空白。",
            "L2": "弹性容器中主轴方向上的对齐方式由专门的对齐属性决定。",
            "L3": "参见 MDN 文档第 2 页关于主轴对齐的说明。",
            "L4": "在容器规则中补回 justify-content: center;"}
    lc = layer_compliance(good, "justify-content", "center")
    assert lc["compliant"] and not lc["leaked"], lc

    bad = dict(good, L1="justify-content 没设置")
    assert layer_compliance(bad, "justify-content")["leaked"] is True

    print("metrics.py 全部自测通过。")
