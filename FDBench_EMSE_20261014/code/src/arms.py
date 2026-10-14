"""
arms.py — 六个对照组（阶梯式消融）

每相邻两组只相差一个组件，因此每一步的增益都可归因：
    A  无检索、纯文本输入                       —— 下界
    B  + 文本检索                     (A->B 检索的贡献)
    C  + 截图作为视觉输入             (B->C 视觉输入的贡献)
    D  + 视觉检索替代文本检索         (C->D 视觉检索的贡献)
    E  + 基础执行验证（渲染是否成功）  (D->E 执行验证的贡献)
    F  + 渲染结构比对与迭代闭环 + 分层反馈 = 完整 M-RAG-Ed
                                       (E->F 渲染验证闭环的贡献)

公平性硬约束（务必遵守，否则结论不成立）：
  - 六组共用同一基座模型、同一温度、同一 max_tokens
  - 六组的知识库内容完全相同，只是索引模态不同
  - 六组的检索条数 k 相同
  - A~D 单轮；E、F 最多 MAX_ITERS 轮，且把轮次开销如实计入效率指标

【本版改动】模型不再返回完整 HTML，只返回“改哪一行、改成什么”。
原因：任务均为单点变异，修复即改一行；让模型吐整份 HTML 会使输出达数千
token，导致极慢、易超时、且 HTML 转义进 JSON 常引发解析失败。改为返回
edits 后，代码侧据行号自动重建完整 HTML，输出量降到几十 token。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

MAX_ITERS = 3
TOP_K = 5

# ==========================================================================
# 统一的输出契约（JSON）
# ==========================================================================

OUTPUT_CONTRACT = """严格只输出一个 JSON 对象，不要任何解释性前后缀，不要 Markdown 代码围栏。
不要返回完整 HTML，只需指出要修改的行及其修改后的内容。字段定义：
{
  "property": "你判定出问题的那一个 CSS 属性名（只写属性名，如 justify-content）",
  "line": 你判定的问题所在行号（整数，基于给出的带行号代码）,
  "region": {"x": 数, "y": 数, "w": 数, "h": 数},   // 截图中异常区域的像素包围盒
  "fixed_css_declaration": "修复后应有的那一条完整声明，如 justify-content: center;",
  "edits": [
    {"line": 行号(整数), "new_text": "该行修改后的完整内容（不含行号前缀）"}
  ],
  "feedback": {
    "L1": "只描述现象，不出现任何 CSS 属性名",
    "L2": "点明相关概念，不出现具体属性取值",
    "L3": "指出证据出处（引用检索到的文档编号或标题）",
    "L4": "给出参考修改"
  }
}
说明：edits 用于精确定位修改。若需新增一行，new_text 用换行符 \\n 承载多行；
若某行应被删除，new_text 置为空字符串。只列出确需改动的行，通常 1-2 行。"""

SYSTEM_PROMPT = """你是一位前端课程的助教。学生提交了一段有问题的 HTML/CSS 代码和运行截图。
你的任务是定位问题、给出修复，并按四个层次组织反馈。修复以“修改哪一行、改成什么”的形式给出。
严禁返回完整的 HTML 源码：只在 edits 中列出确需改动的 1-2 行，绝不要复述整份代码，否则会破坏 JSON 结构。
反馈分层规则必须严格遵守：L1 只讲现象，L2 只讲概念，L3 只讲证据来源，L4 才给出具体修改。
不要在 L1-L3 中泄露答案。请只输出一个合法的 JSON 对象。"""


def number_lines(code: str) -> str:
    return "\n".join(f"{i+1:4d}| {ln}" for i, ln in enumerate(code.split("\n")))


# ==========================================================================
# 从 edits 重建完整 HTML
# ==========================================================================

def apply_edits(original_html: str, edits: list[dict]) -> str:
    """按 edits（行号 + 新内容）重建 HTML。行号 1-based。

    - new_text 为空串 -> 删除该行
    - new_text 含 \\n -> 该行被替换为多行
    - 行号越界的 edit 跳过（模型偶尔给错号）
    """
    lines = original_html.split("\n")
    # 从后往前改，避免行号因增删而错位
    for e in sorted(edits, key=lambda x: -int(x.get("line", 0))):
        ln = int(e.get("line", 0))
        if ln < 1 or ln > len(lines):
            continue
        new_text = e.get("new_text", "")
        if new_text == "":
            del lines[ln - 1]
        else:
            lines[ln - 1: ln] = new_text.split("\n")
    return "\n".join(lines)


# ==========================================================================
# 单次运行的记录
# ==========================================================================

@dataclass
class ArmRun:
    task_id: str
    arm: str
    iters: int = 0
    latency_s: float = 0.0
    in_tokens: int = 0
    out_tokens: int = 0
    retrieved: list[str] = field(default_factory=list)
    parsed: dict = field(default_factory=dict)
    fixed_html_per_iter: list[str] = field(default_factory=list)
    error: str = ""


def parse_json_loose(text: str) -> dict:
    """宽松解析模型输出的 JSON。批量跑时模型偶尔会：加代码围栏、
    把整份 HTML 塞进字段导致内部引号/换行破坏结构、或加前后缀说明。
    这里逐层抢救，尽量不因单点脏输出丢掉整条运行。"""
    t = text.strip()
    # 1) 去掉 markdown 代码围栏
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s*```$", "", t, flags=re.MULTILINE).strip()

    # 2) 截取第一个 { 到最后一个 } 之间（去前后缀说明）
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("未找到 JSON 对象")
    core = t[i:j + 1]

    # 3) 直接解析
    try:
        return json.loads(core)
    except json.JSONDecodeError:
        pass

    # 4) 抢救：模型常把多行 HTML 未转义地塞进字符串值，导致内部裸换行/引号。
    #    先尝试把字符串值内部的裸换行转义掉再解析。
    try:
        repaired = _escape_inner_newlines(core)
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # 5) 最后兜底：只把我们真正需要的字段用正则抠出来，
    #    宁可丢掉 fixed_html 这种大字段，也要保住 edits/property/feedback。
    salvaged = _regex_salvage(core)
    if salvaged:
        return salvaged
    raise ValueError("JSON 解析失败且无法抢救")


def _escape_inner_newlines(s: str) -> str:
    """把 JSON 字符串字面量内部的裸换行/回车转义为 \\n。
    简单状态机：跟踪是否在字符串内、是否被反斜杠转义。"""
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _regex_salvage(s: str) -> dict:
    """从坏 JSON 中尽力抠出关键字段。丢弃无法解析的大字段（如 fixed_html）。"""
    d: dict = {}
    m = re.search(r'"property"\s*:\s*"([^"]*)"', s)
    if m:
        d["property"] = m.group(1)
    m = re.search(r'"line"\s*:\s*(\d+)', s)
    if m:
        d["line"] = int(m.group(1))
    # region
    m = re.search(r'"region"\s*:\s*(\{[^}]*\})', s)
    if m:
        try:
            d["region"] = json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'"fixed_css_declaration"\s*:\s*"([^"]*)"', s)
    if m:
        d["fixed_css_declaration"] = m.group(1)
    # edits：抠出每个 {line,new_text}
    edits = []
    for em in re.finditer(r'\{\s*"line"\s*:\s*(\d+)\s*,\s*"new_text"\s*:\s*'
                          r'"((?:[^"\\]|\\.)*)"\s*\}', s):
        try:
            edits.append({"line": int(em.group(1)),
                          "new_text": json.loads('"' + em.group(2) + '"')})
        except Exception:
            pass
    if edits:
        d["edits"] = edits
    # feedback 四层
    fb = {}
    for lv in ("L1", "L2", "L3", "L4"):
        fm = re.search(rf'"{lv}"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
        if fm:
            try:
                fb[lv] = json.loads('"' + fm.group(1) + '"')
            except Exception:
                fb[lv] = fm.group(1)
    if fb:
        d["feedback"] = fb
    return d


# ==========================================================================
# 提示词构造
# ==========================================================================

def build_prompt(task: dict, arm: str, evidence: Optional[list[dict]] = None,
                 verify_feedback: Optional[str] = None) -> list[dict]:
    parts = [f"学生的问题描述：{task['complaint']}", "",
             "学生当前的代码（带行号）：", "```html",
             number_lines(open_text(task, "student")), "```"]

    if evidence:
        parts += ["", f"检索到的参考资料（共 {len(evidence)} 条，引用时请写出编号）："]
        for i, e in enumerate(evidence, 1):
            parts.append(f"[{i}] {e['title']}\n{e['snippet']}")

    if verify_feedback:
        parts += ["", "上一轮修改经过实际渲染验证，结果如下：", verify_feedback,
                  "请据此继续修正。"]

    parts += ["", OUTPUT_CONTRACT]
    text = "\n".join(parts)

    msg: dict = {"role": "user", "content": text}
    if arm in ("C", "D", "E", "F"):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "第一张为目标效果，第二张为学生当前运行结果。\n" + text},
            {"type": "image", "path": task["target_png"]},
            {"type": "image", "path": task["student_png"]},
        ]}
    return [{"role": "system", "content": SYSTEM_PROMPT}, msg]


def open_text(task: dict, which: str) -> str:
    from pathlib import Path
    base = Path(task["_root"]) / "html" / f"{task['task_id']}_{which}.html"
    return base.read_text(encoding="utf-8")


# ==========================================================================
# 验证器
# ==========================================================================

def verify_basic(fixed_html: str, browser) -> tuple[bool, str]:
    from render import render_and_extract
    r = render_and_extract(fixed_html, None, browser)
    if not r.ok:
        return False, f"页面渲染失败：{r.error[:200]}"
    if len(r.boxes) < 3:
        return False, "页面渲染后几乎没有可见元素，可能结构被破坏。"
    return True, "渲染成功。"


def verify_render_diff(fixed_html: str, target_boxes: list[dict],
                       browser) -> tuple[bool, str]:
    from render import render_and_extract
    from metrics import compare_trees, is_fixed

    r = render_and_extract(fixed_html, None, browser)
    if not r.ok:
        return False, f"页面渲染失败：{r.error[:200]}"

    c = compare_trees(target_boxes, r.boxes)
    if is_fixed(target_boxes, r.boxes):
        return True, "渲染结果与目标完全一致，验证通过。"

    lines = [f"结构比对未通过：{len(c['mismatches'])} 个元素位置或尺寸与目标不符"
             f"（缺失 {c['missing']} 个，多出 {c['extra']} 个，"
             f"结构相似度 {c['struct_sim']:.2f}）。前若干处差异："]
    for m in c["mismatches"][:5]:
        lines.append(
            f"  - 元素 {m['path']}（{m['tag']}）：相对目标向右偏移 {m['dx']:+.0f}px、"
            f"向下偏移 {m['dy']:+.0f}px，尺寸相对差 {m['size_diff']:.1%}")
    return False, "\n".join(lines)


# ==========================================================================
# 主运行循环
# ==========================================================================

def run_arm(task: dict, arm: str, llm: Callable, retriever_text: Callable,
            retriever_visual: Callable, browser) -> ArmRun:
    run = ArmRun(task_id=task["task_id"], arm=arm)
    t0 = time.time()
    try:
        student_html = open_text(task, "student")

        # --- 检索 ---
        evidence = None
        if arm == "A":
            evidence = None
        elif arm in ("B", "C"):
            evidence = retriever_text(task["complaint"], TOP_K)
        else:
            evidence = retriever_visual(task["complaint"], TOP_K,
                                        image_path=task["student_png"])
        if evidence:
            run.retrieved = [e["doc_id"] for e in evidence]

        # --- 生成与验证 ---
        max_iters = MAX_ITERS if arm in ("E", "F") else 1
        vfb = None
        cur_html = student_html
        for it in range(max_iters):
            msgs = build_prompt(task, arm, evidence, vfb)
            text, ti, to = llm(msgs)
            run.in_tokens += ti
            run.out_tokens += to
            run.iters = it + 1
            run.parsed = parse_json_loose(text)

            # 从 edits 重建完整 HTML（供验证器渲染与阶段四复算）
            edits = run.parsed.get("edits", [])
            if edits:
                cur_html = apply_edits(student_html, edits)
            else:
                # 兜底：若模型仍给了 fixed_html，用它；否则维持原状
                cur_html = run.parsed.get("fixed_html", cur_html)
            run.fixed_html_per_iter.append(cur_html)

            if arm == "E":
                ok, vfb = verify_basic(cur_html, browser)
            elif arm == "F":
                ok, vfb = verify_render_diff(cur_html, task["target_boxes"], browser)
            else:
                break
            if ok:
                break
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
    run.latency_s = time.time() - t0
    return run
