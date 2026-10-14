"""
E8 second-generator wrapper: Kimi K2.5

Interface:
    llm(messages) -> (text, in_tokens, out_tokens)

The API-level settings intentionally mirror the frozen Qwen wrapper:
- OpenAI-compatible DashScope endpoint
- temperature = 0.0
- max_tokens = 4096
- enable_thinking = False
- response_format = json_object
- image input via base64 data URL
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path

from openai import OpenAI


MODEL = os.environ.get(
    "KIMI_MODEL",
    "kimi-k2.5",
)

BASE_URL = os.environ.get(
    "KIMI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

TEMPERATURE = 0.0
MAX_TOKENS = 4096
TIMEOUT = 120
MAX_RETRY = 5
RETRY_BASE = 2.0

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client

    if _client is None:
        key = os.environ.get(
            "DASHSCOPE_API_KEY"
        )

        if not key:
            raise RuntimeError(
                "未设置环境变量 DASHSCOPE_API_KEY，请先 export。"
            )

        _client = OpenAI(
            api_key=key,
            base_url=BASE_URL,
            timeout=TIMEOUT,
            max_retries=0,
        )

    return _client


def _img_to_data_url(path: str) -> str:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"图片不存在：{path}"
        )

    mime = (
        mimetypes.guess_type(str(p))[0]
        or
        "image/png"
    )

    b64 = base64.b64encode(
        p.read_bytes()
    ).decode("ascii")

    return (
        f"data:{mime};base64,{b64}"
    )


def _convert_messages(
    messages: list[dict]
) -> list[dict]:

    out = []

    for m in messages:
        content = m["content"]

        if isinstance(content, str):
            out.append({
                "role": m["role"],
                "content": content,
            })
            continue

        conv = []

        for part in content:
            if part.get("type") == "text":
                conv.append({
                    "type": "text",
                    "text": part["text"],
                })

            elif part.get("type") == "image":
                conv.append({
                    "type": "image_url",
                    "image_url": {
                        "url":
                            _img_to_data_url(
                                part["path"]
                            )
                    },
                })

            elif part.get("type") == "image_url":
                conv.append(part)

        out.append({
            "role": m["role"],
            "content": conv,
        })

    return out


def _ensure_json_hint(
    messages: list[dict]
) -> None:

    for m in messages:
        if m["role"] != "system":
            continue

        c = m["content"]

        if (
            isinstance(c, str)
            and
            "json" not in c.lower()
        ):
            m["content"] = (
                c
                +
                "\n请只输出一个合法的 JSON 对象。"
            )

        return


def llm(
    messages: list[dict],
    use_json_mode: bool = True,
) -> tuple[str, int, int]:

    client = _get_client()

    conv = _convert_messages(
        messages
    )

    _ensure_json_hint(
        conv
    )

    kwargs = dict(
        model=MODEL,
        messages=conv,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={
            "enable_thinking": False
        },
    )

    if use_json_mode:
        kwargs["response_format"] = {
            "type": "json_object"
        }

    last_err = None

    for attempt in range(MAX_RETRY):
        try:
            resp = (
                client.chat.completions.create(
                    **kwargs
                )
            )

            text = (
                resp.choices[0]
                .message.content
                or
                ""
            )

            usage = resp.usage

            return (
                text,
                int(usage.prompt_tokens),
                int(usage.completion_tokens),
            )

        except Exception as exc:
            last_err = exc

            msg = str(exc).lower()

            if any(
                x in msg
                for x in (
                    "invalid api key",
                    "unauthorized",
                    "permission",
                    "model not found",
                )
            ):
                raise

            wait = (
                RETRY_BASE
                *
                (2 ** attempt)
            )

            print(
                f"[Kimi retry "
                f"{attempt+1}/{MAX_RETRY}] "
                f"{type(exc).__name__}: "
                f"{str(exc)[:160]} "
                f"— {wait:.0f}s"
            )

            time.sleep(wait)

    raise RuntimeError(
        f"Kimi 调用重试 "
        f"{MAX_RETRY} 次仍失败："
        f"{last_err}"
    )


if __name__ == "__main__":
    print("MODEL =", MODEL)
    print("BASE_URL =", BASE_URL)
    print("TEMPERATURE =", TEMPERATURE)
    print("MAX_TOKENS =", MAX_TOKENS)
