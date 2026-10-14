"""
build_kb.py — 阶段二(1/2)：从 MDN 文档构建"同内容、双模态"知识库

核心原则（手册 4.1 节）：文本索引与视觉索引必须来自同一批文档，
只是索引模态不同，否则比的是语料差异而非模态差异，H1/H2 失效。
因此本脚本对每篇文档同时产出：
  - 文本块（标题层级切分）    -> 供 BM25 + BGE-m3 文本索引
  - 页面图像（markdown->HTML->截图） -> 供 ColPali 视觉索引

子集划分（按 front-matter 的 page-type 自动判定）：
  KB-1 属性参考页  reference/properties/*      (css-property/css-*-property)
  KB-2 布局指南页  guides/*                     (css-module/guide)
  KB-3 属性速查页  由 KB-1 自动生成的"示例+渲染并排"页（强图文型证据）

同时产出 gold 映射：14 个变异属性 -> 其 gold 文档 id 集合，
用于阶段四计算 Doc-Hit（检索真值），在建库阶段即固定。

运行：
    python build_kb.py --mdn ../data/mdn/content --out ../data/kb
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---- 你的 14 个变异算子实际改动的 CSS 属性（与 mutate.py 对应）----
MUTATED_PROPS = [
    "justify-content", "align-items", "display", "flex-direction",
    "margin", "text-align", "box-sizing", "width", "padding",
    "max-width", "position", "overflow", "z-index", "line-height",
    "font-size", "font-weight",
]

# 属性 -> 所属布局机制指南页（用于 gold 兜底：属性页命中或其指南页命中均算对）
PROP_TO_GUIDE = {
    "justify-content": "flexible_box_layout", "align-items": "box_alignment",
    "flex-direction": "flexible_box_layout", "display": "display",
    "margin": "box_model", "padding": "box_model", "box-sizing": "box_sizing",
    "width": "box_model", "max-width": "sizing", "overflow": "overflow",
    "position": "positioned_layout", "z-index": "positioned_layout",
    "text-align": "text", "line-height": "text",
    "font-size": "fonts", "font-weight": "fonts",
}

CSS_ROOT = "files/en-us/web/css"


# --------------------------------------------------------------------------
# front-matter 解析
# --------------------------------------------------------------------------
def parse_front_matter(md_text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", md_text, re.DOTALL)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"^([\w-]+):\s*(.*)$", line)
        if mm:
            fm[mm.group(1)] = mm.group(2).strip()
    return fm


def strip_front_matter(md_text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", md_text, count=1, flags=re.DOTALL)


# --------------------------------------------------------------------------
# 文档发现：优先 reference/properties/<prop>，避免抓到 @media/@font-face 下的同名页
# --------------------------------------------------------------------------
def resolve_prop_dir(css_root: Path, prop: str) -> Path | None:
    primary = css_root / "reference" / "properties" / prop
    if (primary / "index.md").exists():
        return primary
    # 回退：全局搜同名目录，但排除 at-rules 下的描述符同名页
    for d in css_root.rglob(prop):
        if d.is_dir() and (d / "index.md").exists() and "at-rules" not in d.parts:
            return d
    return None


def collect_docs(css_root: Path) -> list[dict]:
    """遍历 reference/properties 与 guides，返回文档记录列表。"""
    docs = []

    # KB-1：属性参考页（全部 properties，不止 14 个，其余作干扰项）
    prop_dir = css_root / "reference" / "properties"
    if prop_dir.exists():
        for d in sorted(prop_dir.iterdir()):
            idx = d / "index.md"
            if not idx.exists():
                continue
            raw = idx.read_text(encoding="utf-8", errors="ignore")
            fm = parse_front_matter(raw)
            docs.append({
                "doc_id": f"kb1:{d.name}",
                "subset": "KB-1",
                "prop": d.name,
                "title": fm.get("title", d.name),
                "slug": fm.get("slug", ""),
                "body_md": strip_front_matter(raw),
                "src": str(idx),
            })

    # KB-2：布局指南页（guides 下全部，图文混排、含示意图）
    guides_dir = css_root / "guides"
    if guides_dir.exists():
        for d in sorted(guides_dir.iterdir()):
            idx = d / "index.md"
            if not idx.exists():
                continue
            raw = idx.read_text(encoding="utf-8", errors="ignore")
            fm = parse_front_matter(raw)
            docs.append({
                "doc_id": f"kb2:{d.name}",
                "subset": "KB-2",
                "prop": "",
                "title": fm.get("title", d.name),
                "slug": fm.get("slug", ""),
                "body_md": strip_front_matter(raw),
                "src": str(idx),
            })
    return docs


# --------------------------------------------------------------------------
# KB-3：属性速查页 —— 从属性页正文抽取代码示例，与其渲染效果并排
# 关键信息落在"渲染图"上而非文字里，是视觉检索的优势区（检验 H2）
# --------------------------------------------------------------------------
def extract_examples(body_md: str, max_examples: int = 2) -> list[str]:
    """抽取 markdown 中的 CSS/HTML 代码块作为速查示例。"""
    blocks = re.findall(r"```(?:css|html)\n(.*?)```", body_md, re.DOTALL)
    out = []
    for b in blocks:
        b = b.strip()
        if 10 < len(b) < 800:
            out.append(b)
        if len(out) >= max_examples:
            break
    return out


def build_kb3_html(prop: str, title: str, examples: list[str]) -> str:
    """把示例代码与其渲染结果并排，生成强图文型速查页 HTML。"""
    cards = []
    for i, ex in enumerate(examples):
        is_css = ("{" in ex and ":" in ex and "<" not in ex)
        if is_css:
            demo = (f"<style>.demo{i} {{ {ex} }}</style>"
                    f'<div class="demo{i}" style="border:1px solid #ccc;padding:8px;'
                    f'min-height:48px">示例文本 A / B / C</div>')
            code = ex
        else:
            demo = ex
            code = ex
        cards.append(
            f'<div style="display:flex;gap:16px;margin:12px 0;'
            f'border-bottom:1px solid #eee;padding-bottom:12px">'
            f'<pre style="flex:1;background:#f6f8fa;padding:10px;'
            f'font:13px monospace;white-space:pre-wrap">{_esc(code)}</pre>'
            f'<div style="flex:1">{demo}</div></div>')
    return (f'<html><head><meta charset="utf-8"></head>'
            f'<body style="font:15px sans-serif;max-width:900px;margin:24px auto">'
            f'<h1 style="font-size:22px">{_esc(title)} — 速查</h1>'
            f'<p style="color:#555">属性 <code>{prop}</code> 的用法示例与渲染效果</p>'
            f'{"".join(cards)}</body></html>')


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# --------------------------------------------------------------------------
# markdown 正文 -> 简单 HTML（用于 KB-1/KB-2 页面截图）
# 不追求完美渲染，只需保留标题层级与段落结构供视觉编码
# --------------------------------------------------------------------------
def md_to_html(title: str, body_md: str) -> str:
    # 去掉 MDN 宏 {{...}} 与 sidebar 等噪声
    txt = re.sub(r"\{\{.*?\}\}", "", body_md)
    txt = re.sub(r"^\s*sidebar:.*$", "", txt, flags=re.MULTILINE)
    lines = []
    for ln in txt.splitlines():
        h = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if h:
            lv = len(h.group(1))
            lines.append(f"<h{lv}>{_esc(h.group(2))}</h{lv}>")
        elif ln.strip():
            lines.append(f"<p>{_esc(ln.strip())}</p>")
    body = "\n".join(lines[:120])  # 截断超长页，控制截图高度
    return (f'<html><head><meta charset="utf-8"></head>'
            f'<body style="font:15px/1.6 sans-serif;max-width:820px;'
            f'margin:24px auto;color:#111">'
            f'<h1>{_esc(title)}</h1>{body}</body></html>')


# --------------------------------------------------------------------------
# 文本切块：按标题层级切，供 BM25 + 稠密检索
# --------------------------------------------------------------------------
def chunk_by_heading(doc: dict, max_chars: int = 600) -> list[dict]:
    txt = re.sub(r"\{\{.*?\}\}", "", doc["body_md"])
    parts = re.split(r"\n(?=#{1,4}\s)", txt)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if len(part) < 20:
            continue
        for j in range(0, len(part), max_chars):
            seg = part[j:j + max_chars]
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#c{i}_{j//max_chars}",
                "doc_id": doc["doc_id"],
                "subset": doc["subset"],
                "title": doc["title"],
                "text": seg,
            })
    return chunks


# --------------------------------------------------------------------------
# gold 映射：变异属性 -> gold 文档集合（属性页 + 其指南页 + 其速查页）
# --------------------------------------------------------------------------
def build_gold(docs: list[dict]) -> dict:
    have = {d["doc_id"] for d in docs}
    have_kb3 = {d["doc_id"] for d in docs if d["subset"] == "KB-3"}
    gold = {}
    for prop in MUTATED_PROPS:
        g = set()
        if f"kb1:{prop}" in have:
            g.add(f"kb1:{prop}")
        guide = PROP_TO_GUIDE.get(prop)
        if guide and f"kb2:{guide}" in have:
            g.add(f"kb2:{guide}")
        if f"kb3:{prop}" in have_kb3:
            g.add(f"kb3:{prop}")
        gold[prop] = sorted(g)
    return gold


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main(mdn: Path, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    css_root = mdn / CSS_ROOT
    if not css_root.exists():
        raise SystemExit(f"未找到 CSS 文档根目录：{css_root}")

    (out / "img").mkdir(parents=True, exist_ok=True)
    docs = collect_docs(css_root)
    print(f"收集文档：KB-1 {sum(d['subset']=='KB-1' for d in docs)} 篇，"
          f"KB-2 {sum(d['subset']=='KB-2' for d in docs)} 篇")

    # 生成 KB-3 速查页（仅对 14 个变异属性，控制规模且聚焦评测）
    kb1_by_prop = {d["prop"]: d for d in docs if d["subset"] == "KB-1"}
    kb3_docs = []
    for prop in MUTATED_PROPS:
        d = kb1_by_prop.get(prop)
        if not d:
            continue
        examples = extract_examples(d["body_md"])
        if not examples:
            continue
        kb3_docs.append({
            "doc_id": f"kb3:{prop}", "subset": "KB-3", "prop": prop,
            "title": f"{d['title']} 速查", "slug": "",
            "_html": build_kb3_html(prop, d["title"], examples),
            "src": "generated",
        })
    docs.extend(kb3_docs)
    print(f"生成 KB-3 速查页 {len(kb3_docs)} 篇")

    # 渲染页面图像 + 落盘文本块
    all_chunks = []
    meta = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--hide-scrollbars"])
        page = browser.new_page(viewport={"width": 900, "height": 1200},
                                device_scale_factor=1)
        for d in docs:
            html = d.get("_html") or md_to_html(d["title"], d["body_md"])
            img_rel = f"img/{d['doc_id'].replace(':', '_')}.png"
            try:
                page.set_content(html, wait_until="load", timeout=15000)
                page.wait_for_timeout(120)
                page.screenshot(path=str(out / img_rel), full_page=True)
            except Exception as e:
                print(f"  [跳过截图] {d['doc_id']}: {e}")
                img_rel = ""
            if "body_md" in d:
                all_chunks.extend(chunk_by_heading(d))
            meta.append({
                "doc_id": d["doc_id"], "subset": d["subset"],
                "prop": d.get("prop", ""), "title": d["title"],
                "image": img_rel, "src": d.get("src", ""),
            })
        browser.close()

    gold = build_gold(docs)

    (out / "docs.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in meta),
        encoding="utf-8")
    (out / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks),
        encoding="utf-8")
    (out / "gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

    # 自查：哪些属性没有任何 gold（会导致该类任务 Doc-Hit 恒为 0）
    empty = [p for p, g in gold.items() if not g]
    print(f"\n知识库完成：文档 {len(meta)} 篇，文本块 {len(all_chunks)} 个，"
          f"图像 {sum(1 for m in meta if m['image'])} 张")
    print(f"gold 映射覆盖 {len(gold)-len(empty)}/{len(gold)} 个属性")
    if empty:
        print(f"  [警告] 以下属性无 gold 文档，请检查：{empty}")
    print(f"输出目录：{out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdn", type=Path, required=True,
                    help="MDN content 仓库根目录（含 files/en-us/web/css）")
    ap.add_argument("--out", type=Path, default=Path("../data/kb"))
    a = ap.parse_args()
    main(a.mdn, a.out)
