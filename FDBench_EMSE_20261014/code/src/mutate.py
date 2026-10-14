"""
mutate.py — 受控变异算子引擎

作用：对一份"正确"的 HTML/CSS 源码施加单点变异，生成"有缺陷"的版本，
      同时自动产出真值标注（被改动的属性、行号、所属类别）。

设计原则（三条，缺一不可）：
  1) 语法有效：变异后仍能正常渲染，制造"看起来对但结果不对"的情形；
  2) 视觉可见：变异必须引起可观测的渲染差异（由 build_tasks.py 过滤）；
  3) 单点唯一：一次只改一处，保证缺陷定位的真值无歧义。

用法：
    from mutate import MUTATORS, apply_mutation
    result = apply_mutation(html_text, "drop_justify_content", rng)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, asdict
from typing import Callable, Optional


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class MutationResult:
    """一次成功变异的完整记录，直接作为任务真值落盘。"""
    op_id: str                 # 算子标识
    category: str              # 类别：layout / boxmodel / stacking / text
    mutated_html: str          # 变异后的源码
    gt_property: str           # 真值：被改动的 CSS 属性名（主定位指标）
    gt_line: int               # 真值：被改动的行号（1-based）
    gt_before: str             # 原始声明文本
    gt_after: str              # 变异后声明文本（删除则为空串）
    symptom_hint: str          # 症状类别，用于生成学习者口吻的自然语言诉求

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mutated_html")   # 源码单独存文件，不进 JSON
        return d


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------

# 匹配一条 CSS 声明：属性名 : 值 ;
# 注意：值必须用贪婪匹配并显式排除换行，否则会在第一个字符处提前截断，
#       导致 display:flex 这类"按值筛选"的算子全部失效。
_DECL = (r"(?P<indent>[ \t]*)(?P<prop>{prop})\s*:\s*"
         r"(?P<val>[^;{{}}\n]+?)[ \t]*(?P<semi>;|(?=[\n}}]))")


def _find_decls(html: str, prop: str, value_pat: str = r".*") -> list[re.Match]:
    """找出源码中所有匹配 `prop: value` 的声明。"""
    pat = re.compile(_DECL.format(prop=re.escape(prop)), re.IGNORECASE)
    out = []
    for m in pat.finditer(html):
        if re.fullmatch(value_pat, m.group("val").strip(), re.IGNORECASE):
            out.append(m)
    return out


def _line_of(html: str, pos: int) -> int:
    """字符偏移 -> 1-based 行号。"""
    return html.count("\n", 0, pos) + 1


def _replace_span(html: str, m: re.Match, new_text: str) -> str:
    return html[: m.start()] + new_text + html[m.end():]


def _pick(rng: random.Random, seq: list):
    return seq[rng.randrange(len(seq))] if seq else None


def _delete_decl(html: str, m: re.Match) -> str:
    """整条删除声明，保留缩进以免行号错位过多。"""
    return _replace_span(html, m, m.group("indent"))


def _rewrite_value(html: str, m: re.Match, new_val: str) -> str:
    return _replace_span(
        html,
        m,
        f"{m.group('indent')}{m.group('prop')}: {new_val}{m.group('semi')}",
    )


# --------------------------------------------------------------------------
# 算子实现
# 每个算子签名：(html, rng) -> Optional[MutationResult]
# 返回 None 表示该源码不适用此算子（没有可改的目标）。
# --------------------------------------------------------------------------

def _make_drop_op(op_id: str, category: str, prop: str,
                  symptom: str, value_pat: str = r".*") -> Callable:
    """生成"删除某条声明"型算子。"""

    def _op(html: str, rng: random.Random) -> Optional[MutationResult]:
        m = _pick(rng, _find_decls(html, prop, value_pat))
        if m is None:
            return None
        return MutationResult(
            op_id=op_id,
            category=category,
            mutated_html=_delete_decl(html, m),
            gt_property=prop,
            gt_line=_line_of(html, m.start()),
            gt_before=m.group(0).strip(),
            gt_after="",
            symptom_hint=symptom,
        )

    return _op


def _make_swap_op(op_id: str, category: str, prop: str, mapping: dict,
                  symptom: str) -> Callable:
    """生成"替换属性值"型算子。mapping: 原值 -> 新值。"""

    def _op(html: str, rng: random.Random) -> Optional[MutationResult]:
        cands = [m for m in _find_decls(html, prop)
                 if m.group("val").strip().lower() in mapping]
        m = _pick(rng, cands)
        if m is None:
            return None
        old = m.group("val").strip().lower()
        new = mapping[old]
        return MutationResult(
            op_id=op_id,
            category=category,
            mutated_html=_rewrite_value(html, m, new),
            gt_property=prop,
            gt_line=_line_of(html, m.start()),
            gt_before=m.group(0).strip(),
            gt_after=f"{m.group('prop')}: {new};",
            symptom_hint=symptom,
        )

    return _op


def _op_drop_auto_margin(html: str, rng: random.Random) -> Optional[MutationResult]:
    """margin: 0 auto -> margin: 0，破坏块级元素水平居中。"""
    cands = [m for m in _find_decls(html, "margin")
             if "auto" in m.group("val").lower()]
    m = _pick(rng, cands)
    if m is None:
        return None
    new_val = re.sub(r"\bauto\b", "0", m.group("val"), flags=re.IGNORECASE)
    return MutationResult(
        op_id="drop_auto_margin", category="layout",
        mutated_html=_rewrite_value(html, m, new_val),
        gt_property="margin", gt_line=_line_of(html, m.start()),
        gt_before=m.group(0).strip(), gt_after=f"margin: {new_val};",
        symptom_hint="horizontal_shift",
    )


def _op_percent_to_px(html: str, rng: random.Random) -> Optional[MutationResult]:
    """width: NN% -> width: NNpx，制造响应式失效（数值不变、单位改变）。"""
    cands = [m for m in _find_decls(html, "width")
             if re.fullmatch(r"\d{1,3}%", m.group("val").strip())]
    m = _pick(rng, cands)
    if m is None:
        return None
    num = m.group("val").strip().rstrip("%")
    new_val = f"{num}px"
    return MutationResult(
        op_id="percent_to_px", category="boxmodel",
        mutated_html=_rewrite_value(html, m, new_val),
        gt_property="width", gt_line=_line_of(html, m.start()),
        gt_before=m.group(0).strip(), gt_after=f"width: {new_val};",
        symptom_hint="size_wrong",
    )


def _op_inject_overflow_hidden(html: str, rng: random.Random) -> Optional[MutationResult]:
    """在某个带 height 的规则中注入 overflow: hidden，制造内容截断。"""
    m = _pick(rng, _find_decls(html, "height"))
    if m is None:
        return None
    injected = (f"{m.group(0)}\n{m.group('indent')}overflow: hidden;")
    return MutationResult(
        op_id="inject_overflow_hidden", category="stacking",
        mutated_html=_replace_span(html, m, injected),
        gt_property="overflow", gt_line=_line_of(html, m.start()) + 1,
        gt_before="", gt_after="overflow: hidden;",
        symptom_hint="content_clipped",
    )


# --------------------------------------------------------------------------
# 算子注册表（14 个，覆盖 4 个类别）
# --------------------------------------------------------------------------

MUTATORS: dict[str, Callable] = {
    # --- 布局类 layout ---
    "drop_justify_content": _make_drop_op(
        "drop_justify_content", "layout", "justify-content", "horizontal_shift"),
    "drop_align_items": _make_drop_op(
        "drop_align_items", "layout", "align-items", "vertical_shift"),
    "flex_to_block": _make_swap_op(
        "flex_to_block", "layout", "display", {"flex": "block"}, "layout_collapse"),
    "flip_flex_direction": _make_swap_op(
        "flip_flex_direction", "layout", "flex-direction",
        {"row": "column", "column": "row"}, "layout_collapse"),
    "drop_auto_margin": _op_drop_auto_margin,
    "drop_text_align": _make_drop_op(
        "drop_text_align", "layout", "text-align", "horizontal_shift",
        value_pat=r"center|right"),

    # --- 盒模型类 boxmodel ---
    "drop_border_box": _make_drop_op(
        "drop_border_box", "boxmodel", "box-sizing", "size_wrong"),
    "percent_to_px": _op_percent_to_px,
    "drop_padding": _make_drop_op(
        "drop_padding", "boxmodel", "padding", "spacing_wrong"),
    "drop_max_width": _make_drop_op(
        "drop_max_width", "boxmodel", "max-width", "size_wrong"),

    # --- 层叠与可见性类 stacking ---
    "drop_position_relative": _make_swap_op(
        "drop_position_relative", "stacking", "position",
        {"relative": "static"}, "anchor_lost"),
    "inject_overflow_hidden": _op_inject_overflow_hidden,
    "drop_zindex": _make_drop_op(
        "drop_zindex", "stacking", "z-index", "occlusion"),

    # --- 文本与间距类 text ---
    "shrink_font_size": _make_swap_op(
        "shrink_font_size", "text", "font-size",
        {"16px": "11px", "15px": "10px", "14px": "10px", "18px": "12px",
         "20px": "13px", "1rem": "0.7rem", "1.1rem": "0.75rem",
         "1.2rem": "0.8rem", "1.25rem": "0.8rem", "1.5rem": "1rem"},
        "spacing_wrong"),

    # 删除粗体：字宽收缩导致同一文本的包围盒宽度变化，行内元素随之位移
    "drop_font_weight": _make_drop_op(
        "drop_font_weight", "text", "font-weight", "spacing_wrong",
        value_pat=r"bold|[6-9]00"),
    "drop_line_height": _make_drop_op(
        "drop_line_height", "text", "line-height", "spacing_wrong"),
}

CATEGORIES = ["layout", "boxmodel", "stacking", "text"]


def apply_mutation(html: str, op_id: str, rng: random.Random) -> Optional[MutationResult]:
    if op_id not in MUTATORS:
        raise KeyError(f"未知算子: {op_id}")
    return MUTATORS[op_id](html, rng)


def applicable_ops(html: str, rng: random.Random) -> list[str]:
    """探测某份源码上所有可用的算子（用于均衡采样）。"""
    ok = []
    for op_id in MUTATORS:
        probe = random.Random(0)          # 固定探测种子，只判断有无目标
        if MUTATORS[op_id](html, probe) is not None:
            ok.append(op_id)
    return ok


# --------------------------------------------------------------------------
# 学习者口吻的自然语言诉求模板
# 关键设计：只描述"症状"，不出现属性名 —— 模拟新手表达，避免答案泄漏
# --------------------------------------------------------------------------

SYMPTOM_TEMPLATES: dict[str, list[str]] = {
    "horizontal_shift": [
        "这块内容跑到左边去了，本来应该在中间的，不知道为什么。",
        "页面里那一栏没有居中，一直贴着边，我改了半天没效果。",
    ],
    "vertical_shift": [
        "这几个元素上下没对齐，看起来高低不平。",
        "一行里的东西垂直方向乱七八糟的，没在一条线上。",
    ],
    "layout_collapse": [
        "本来应该横着排的几个块，现在全叠成一列了。",
        "布局整个散掉了，元素的排列方向不对。",
    ],
    "size_wrong": [
        "这个区域宽度不对，窗口变大以后它也不跟着变。",
        "这块比预期的窄很多，右边空了一大片。",
    ],
    "spacing_wrong": [
        "文字和边框贴得太近了，看着很挤。",
        "间距不对，元素之间挤在一起了。",
    ],
    "content_clipped": [
        "下面的内容被截断了，滚动也看不到。",
        "这个框里的字显示不全，后面的部分没了。",
    ],
    "anchor_lost": [
        "里面那个元素跑到页面很奇怪的位置去了，不在它该在的框里。",
        "定位完全错了，元素飞到别的地方去了。",
    ],
    "occlusion": [
        "两个元素叠在一起了，上面那个把下面的挡住了。",
        "层次不对，本来该在上面的现在被盖住了。",
    ],
}


def make_complaint(symptom_hint: str, rng: random.Random) -> str:
    tpl = SYMPTOM_TEMPLATES.get(symptom_hint, ["页面显示得不对，和设计稿不一样。"])
    return tpl[rng.randrange(len(tpl))]


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """<html>
<head><style>
.container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 80%;
    margin: 0 auto;
    box-sizing: border-box;
    padding: 20px;
}
.card { position: relative; z-index: 5; line-height: 1.6; height: 200px; }
</style></head>
<body><div class="container"><div class="card">hello</div></div></body>
</html>"""

    rng = random.Random(42)
    ops = applicable_ops(SAMPLE, rng)
    print(f"可用算子 {len(ops)}/{len(MUTATORS)}: {ops}\n")

    for op_id in ops:
        r = apply_mutation(SAMPLE, op_id, random.Random(1))
        assert r is not None
        assert r.mutated_html != SAMPLE, f"{op_id} 未产生改动"
        assert 1 <= r.gt_line <= SAMPLE.count("\n") + 2
        print(f"[{r.category:9s}] {r.op_id:24s} L{r.gt_line:<3d} "
              f"{r.gt_before!r} -> {r.gt_after!r}")
        print(f"{'':13s}诉求: {make_complaint(r.symptom_hint, rng)}")

    print("\n自测通过。")
