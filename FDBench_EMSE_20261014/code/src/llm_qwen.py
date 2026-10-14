"""
llm_qwen.py — 通义千问 Qwen3.5-Plus 的 llm() 实现

对接方式：阿里云百炼 DashScope 的 OpenAI 兼容端点。
Qwen3.5-Plus 支持图像输入（C~F 组传截图）、纯文本输出（本实验只需模型吐 JSON）。

提供给 arms.py 的接口：
    llm(messages) -> (text, in_tokens, out_tokens)

messages 为 arms.py.build_prompt 产出的结构；其中图像项形如
    {"type": "image", "path": "/abs/path/xxx.png"}
本模块负责把它转成 DashScope 兼容的 image_url + base64 格式。

环境变量（你自己 export，别写进代码）：
    export DASHSCOPE_API_KEY="sk-你的key"
    可选：export QWEN_MODEL="qwen3.5-plus"   # 若百炼上的确切 id 不同，用它覆盖
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path

from openai import OpenAI

# --- 全局配置（六组共用，保证公平性）------------------------------------
MODEL = os.environ.get("QWEN_MODEL", "qwen3.5-plus")
BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
TEMPERATURE = 0.0          # 硬约束：六组统一 0，不得单独调
MAX_TOKENS = 4096          # 硬约束：六组统一
TIMEOUT = 120              # 单次请求超时（秒）
MAX_RETRY = 5              # 失败重试次数
RETRY_BASE = 2.0           # 指数退避基数（秒）

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise RuntimeError(
                "未设置环境变量 DASHSCOPE_API_KEY，请先 export。")
        _client = OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT,
                         max_retries=0)   # 禁用库自带重试，只用本模块的手动退避
    return _client


def _img_to_data_url(path: str) -> str:
    """本地图片 -> data URL（base64），供 OpenAI 兼容的 image_url 字段使用。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _convert_messages(messages: list[dict]) -> list[dict]:
    """把 arms.py 的 messages 转成 DashScope OpenAI 兼容格式。

    - 纯文本 content 原样保留
    - list 型 content：{"type":"text",...} 保留；
      {"type":"image","path":p} -> {"type":"image_url","image_url":{"url":data_url}}
    """
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        conv = []
        for part in content:
            if part.get("type") == "text":
                conv.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                conv.append({
                    "type": "image_url",
                    "image_url": {"url": _img_to_data_url(part["path"])},
                })
            elif part.get("type") == "image_url":
                conv.append(part)              # 已是目标格式
        out.append({"role": m["role"], "content": conv})
    return out


def _ensure_json_hint(messages: list[dict]) -> None:
    """JSON mode 的兼容要求：提示中须出现 'json' 字样。arms 的契约已含，
    但以防万一，在 system 消息尾部补一句（幂等）。"""
    for m in messages:
        if m["role"] == "system":
            c = m["content"]
            if isinstance(c, str) and "json" not in c.lower():
                m["content"] = c + "\n请只输出一个合法的 JSON 对象。"
            return


def llm(messages: list[dict], use_json_mode: bool = True) -> tuple[str, int, int]:
    """调用 Qwen3.5-Plus，返回 (文本, 输入token, 输出token)。

    带指数退避重试：限流(429)、超时、5xx 都会重试；重试耗尽则抛出。
    """
    client = _get_client()
    conv = _convert_messages(messages)
    _ensure_json_hint(conv)

    kwargs = dict(model=MODEL, messages=conv,
                  temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                  # 关键：关闭 Qwen3 思考模式。开启时模型会先长篇内部推理，
                  # 使简单请求从秒级劣化到分钟级（实测 110s→1.6s），且污染
                  # 六组对比的归因、烧掉大量思考 token。本实验必须关闭。
                  extra_body={"enable_thinking": False})
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            return text, int(u.prompt_tokens), int(u.completion_tokens)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # 不可重试的错误（如鉴权、模型名错）直接抛，别浪费退避
            if any(k in msg for k in ("invalid api key", "not found",
                                      "unauthorized", "permission")):
                raise
            wait = RETRY_BASE * (2 ** attempt)
            print(f"    [llm 重试 {attempt+1}/{MAX_RETRY}] {type(e).__name__}: "
                  f"{str(e)[:120]} — {wait:.0f}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"llm 调用重试 {MAX_RETRY} 次仍失败：{last_err}")


# --------------------------------------------------------------------------
# 冒烟测试：确认 key、模型名、图像输入、JSON 输出都通
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    print(f"模型：{MODEL}　端点：{BASE_URL}")

    # 1) 纯文本 + JSON mode
    msgs = [{"role": "system", "content": "你是助手，只输出 JSON。"},
            {"role": "user", "content": '返回 {"ok": true, "n": 3} 这个 JSON。'}]
    txt, ti, to = llm(msgs)
    print(f"\n[文本测试] 输出={txt!r}  tokens in/out={ti}/{to}")

    # 2) 图像输入（传一张任意本地图确认能收图）
    if len(sys.argv) > 1:
        img = sys.argv[1]
        msgs = [{"role": "system", "content": "你是助手，只输出 JSON。"},
                {"role": "user", "content": [
                    {"type": "text", "text": '用 {"desc": "..."} 描述这张图。'},
                    {"type": "image", "path": img},
                ]}]
        txt, ti, to = llm(msgs)
        print(f"[图像测试] 输出={txt[:200]!r}  tokens in/out={ti}/{to}")
    else:
        print("\n（传一个图片路径作参数可测图像输入，"
              "如：python llm_qwen.py ../data/tasks/img/xxx.png）")
    print("\n冒烟测试通过。")
