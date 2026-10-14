"""
rebuild_gold_kb2.py — 用 MDN 属性页的真实链接结构，客观重建 KB-2 gold

原 gold 的 KB-2 部分是手工猜测的属性→指南页映射，既不准又主观（实测命中 0）。
本脚本改为客观定义：每个属性的 KB-2 gold = 该属性参考页正文中实际链接到的
Guides 指南页。这是 MDN 编辑建立的关联，非主观判断，可辩护、可复现。

做法：
  1. 读每个 KB-1 属性页源文件，抽取正文里 /Web/CSS/Guides/<页>/... 的链接
  2. 取 Guides 后第一段作为指南页名，小写化，对齐 docs.jsonl 里的真实 kb2 id
  3. 只保留 docs.jsonl 中真实存在的 kb2 页
  4. 重写 gold.json 的 KB-2 部分（KB-1、KB-3 保持不变），另存 gold_v2.json

运行：
    python rebuild_gold_kb2.py --kb ../data/kb
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# 抽取正文中指向 Guides 指南页的链接，取 Guides 后第一段
GUIDE_LINK = re.compile(r"/Web/CSS/Guides/([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


# 跨属性通用的指南页——几乎每个属性页都会顺带链接，非专属证据，排除出 gold
GENERIC_GUIDES = {
    "cascade", "cascading_variables", "display", "syntax", "values_and_units",
    "css_values_and_units", "conditional_rules", "comments",
    "shorthand_properties", "inheritance",
    "anchor_positioning",  # 新特性页，被大量属性顺带链接，非专属
}


def extract_guide_ids(md_text: str, real_kb2: set[str],
                      self_guide: str | None = None) -> list[str]:
    """从属性页正文抽出它链接到的、真实存在且非通用的 KB-2 指南页 id。

    self_guide：若属性名本身就是某个指南页（如 overflow），优先纳入。
    只取频次最高的前 2 个专属指南页，避免顺带提及的页混入。
    """
    found = Counter()
    for m in GUIDE_LINK.finditer(md_text):
        gid = m.group(1).lower()          # Flexible_box_layout -> flexible_box_layout
        if gid in real_kb2 and gid not in GENERIC_GUIDES:
            found[gid] += 1
    ranked = [g for g, _ in found.most_common()]
    # 属性同名指南页存在且被链接则置顶（如 overflow 属性 → overflow 指南页）
    if self_guide and self_guide in found:
        ranked = [self_guide] + [g for g in ranked if g != self_guide]
    return ranked[:2]


def main(kb: Path) -> None:
    docs = load_jsonl(kb / "docs.jsonl")
    gold = json.loads((kb / "gold.json").read_text(encoding="utf-8"))

    # 真实存在的 KB-2 id 集合
    real_kb2 = {d["doc_id"].split(":")[1] for d in docs
                if d["doc_id"].startswith("kb2:")}
    # KB-1 属性页的源文件路径
    kb1_src = {d["prop"]: d.get("src", "") for d in docs
               if d["doc_id"].startswith("kb1:") and d.get("prop")}

    new_gold = {}
    report = []
    for prop, gold_ids in gold.items():
        # 保留原有的 KB-1 与 KB-3（客观：属性专属）
        kept = [g for g in gold_ids if g.split(":")[0] in ("kb1", "kb3")]
        # 重建 KB-2：从属性页链接抽取
        kb2_ids = []
        src = kb1_src.get(prop, "")
        if src and Path(src).exists():
            txt = Path(src).read_text(encoding="utf-8", errors="ignore")
            # 属性名可能本身就是指南页名（overflow、display 等）
            self_guide = prop.replace("-", "_").lower()
            self_guide = self_guide if self_guide in real_kb2 else None
            kb2_ids = [f"kb2:{g}"
                       for g in extract_guide_ids(txt, real_kb2, self_guide)]
        new_gold[prop] = kept + kb2_ids
        report.append((prop, len(kept), kb2_ids))

    # 落盘
    (kb / "gold_v2.json").write_text(
        json.dumps(new_gold, ensure_ascii=False, indent=2), encoding="utf-8")

    # 报告
    print("重建 KB-2 gold（基于属性页真实链接）：\n")
    print(f"  {'属性':<18}{'KB1+KB3':<10}{'新KB-2 gold'}")
    n_with_kb2 = 0
    for prop, n_kept, kb2 in report:
        if kb2:
            n_with_kb2 += 1
        kb2_str = ", ".join(x.replace("kb2:", "") for x in kb2) or "（无）"
        print(f"  {prop:<18}{n_kept:<10}{kb2_str}")
    print(f"\n{n_with_kb2}/{len(report)} 个属性获得了 KB-2 gold")
    print(f"新 gold 写入 {kb/'gold_v2.json'}（原 gold.json 未改动）")
    print("\n下一步：用 gold_v2.json 重跑 finalize 与 evaluate（把 --kb 指向的 gold 换成 v2，"
          "或直接把 gold_v2.json 改名覆盖 gold.json 后重跑）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, default=Path("../data/kb"))
    a = ap.parse_args()
    main(a.kb)
