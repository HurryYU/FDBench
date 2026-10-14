"""
render.py — 无头渲染与元素包围盒提取

提供两个核心能力：
  1) render(html) -> 截图 PNG
  2) extract_boxes(html) -> 元素包围盒树（结构级验证的基础）

包围盒树是本实验"结构比对"判据的数据基础。相比像素比对，它对字体、
DPI、抗锯齿不敏感，且差异可直接转写为教学语言（见手册 6.3 节）。

依赖：playwright（需先执行 `playwright install chromium`）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 统一视口。所有渲染必须使用同一视口与缩放，否则跨组不可比。
VIEWPORT = {"width": 1280, "height": 960}
DEVICE_SCALE = 1
# 关闭动画与字体平滑差异来源
INIT_CSS = """
*, *::before, *::after {
  animation: none !important;
  transition: none !important;
  caret-color: transparent !important;
}
"""

# 在页面内执行：为每个元素生成稳定的 DOM 路径 + 包围盒
_EXTRACT_JS = r"""
() => {
  function domPath(el) {
    const parts = [];
    while (el && el.nodeType === 1 && el.tagName !== 'HTML') {
      const parent = el.parentElement;
      if (!parent) break;
      const sibs = Array.from(parent.children).filter(c => c.tagName === el.tagName);
      const idx = sibs.indexOf(el) + 1;
      parts.unshift(el.tagName.toLowerCase() + '[' + idx + ']');
      el = parent;
    }
    return parts.join('/');
  }
  const out = [];
  const all = document.querySelectorAll('body *');
  for (const el of all) {
    const r = el.getBoundingClientRect();
    // 跳过完全不可见元素：它们的位移没有教学意义，且会引入噪声
    if (r.width === 0 && r.height === 0) continue;
    const cs = window.getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    out.push({
      path: domPath(el),
      tag: el.tagName.toLowerCase(),
      x: Math.round((r.x + window.scrollX) * 100) / 100,
      y: Math.round((r.y + window.scrollY) * 100) / 100,
      w: Math.round(r.width * 100) / 100,
      h: Math.round(r.height * 100) / 100,
      text: (el.childElementCount === 0 ? (el.textContent || '').trim().slice(0, 80) : '')
    });
  }
  return out;
}
"""


@dataclass
class RenderResult:
    png_path: Optional[str]
    boxes: list[dict]
    ok: bool
    error: str = ""


def _new_page(pw_browser):
    page = pw_browser.new_page(viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE)
    page.add_init_script("document.addEventListener('DOMContentLoaded',()=>{"
                         "const s=document.createElement('style');"
                         f"s.textContent={json.dumps(INIT_CSS)};"
                         "document.head.appendChild(s);});")
    return page


def render_and_extract(html: str, out_png: Optional[Path], browser,
                       timeout_ms: int = 15000) -> RenderResult:
    """渲染一份 HTML，返回截图路径与包围盒列表。"""
    page = _new_page(browser)
    try:
        page.set_content(html, wait_until="load", timeout=timeout_ms)
        page.wait_for_timeout(300)          # 等待字体与布局稳定
        boxes = page.evaluate(_EXTRACT_JS)
        png = None
        if out_png is not None:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out_png), full_page=True)
            png = str(out_png)
        return RenderResult(png_path=png, boxes=boxes, ok=True)
    except Exception as e:                  # 渲染失败也是一种有效观测
        return RenderResult(png_path=None, boxes=[], ok=False, error=str(e))
    finally:
        page.close()


def batch_render(items: list[tuple[str, Optional[Path]]]) -> list[RenderResult]:
    """批量渲染。复用同一个浏览器实例，显著快于逐个启动。"""
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-device-scale-factor=1",
                                           "--hide-scrollbars"])
        try:
            for html, png in items:
                results.append(render_and_extract(html, png, browser))
        finally:
            browser.close()
    return results


def crop_region(png_path: str, bbox: dict, out_path: str, pad: int = 24) -> str:
    """按包围盒裁剪截图局部，用于给 VLM 提供聚焦视图（可选增强）。"""
    from PIL import Image

    im = Image.open(png_path)
    x0 = max(0, int(bbox["x"]) - pad)
    y0 = max(0, int(bbox["y"]) - pad)
    x1 = min(im.width, int(bbox["x"] + bbox["w"]) + pad)
    y1 = min(im.height, int(bbox["y"] + bbox["h"]) + pad)
    im.crop((x0, y0, x1, y1)).save(out_path)
    return out_path
