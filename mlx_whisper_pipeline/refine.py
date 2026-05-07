"""素起こしテキストを Claude で整文する.

Engine は 2 種類:

- ``claude-code`` (default): subprocess で ``claude -p`` を呼ぶ。Claude Code subscription
  で動作するため Anthropic API key 不要。Claude Code が PATH 上にインストールされている
  必要がある。
- ``api``: Anthropic SDK を使う。``ANTHROPIC_API_KEY`` が必要。CI / 別マシン上で
  実行する用途向け。
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Literal

from .prompts import PROMPT_MAP

DEFAULT_API_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_ENGINE: "Engine" = "claude-code"

Format = Literal["minutes", "summary", "extract"]
Engine = Literal["claude-code", "api"]


def refine(
    raw_text: str,
    format: Format = "minutes",
    engine: Engine = DEFAULT_ENGINE,
    model: str = DEFAULT_API_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Any | None = None,
) -> str:
    """素起こしテキストを Claude で整文・要約・構造化する.

    Parameters
    ----------
    raw_text : str
        素起こしテキスト（mlx-whisper 出力など）。
    format : "minutes" | "summary" | "extract"
        - ``minutes``: 議事録 Markdown
        - ``summary``: 300〜500 字 plain text 要約
        - ``extract``: 決定事項 / 次アクション / 質問を JSON 抽出
    engine : "claude-code" | "api"
        - ``claude-code``: ``claude -p`` を subprocess で呼ぶ（API key 不要）
        - ``api``: Anthropic SDK を使う（``ANTHROPIC_API_KEY`` が必要）
    model : str
        ``engine="api"`` のときに使う Anthropic モデル ID。
        ``engine="claude-code"`` のときは無視される。
    max_tokens : int
        ``engine="api"`` のときの最大出力トークン数。
    client : Any | None
        ``engine="api"`` のとき差し替え可能な Anthropic クライアント。
        None なら ``ANTHROPIC_API_KEY`` から自動生成。

    Returns
    -------
    str
        Claude の出力テキスト。
    """
    if format not in PROMPT_MAP:
        raise ValueError(f"unknown format: {format!r} (expected one of {list(PROMPT_MAP)})")
    if not raw_text.strip():
        raise ValueError("raw_text is empty")

    prompt = PROMPT_MAP[format].format(text=raw_text)

    if engine == "claude-code":
        return _refine_via_claude_code(prompt)
    if engine == "api":
        return _refine_via_api(prompt, model=model, max_tokens=max_tokens, client=client)
    raise ValueError(f"unknown engine: {engine!r} (expected 'claude-code' or 'api')")


def _refine_via_claude_code(prompt: str) -> str:
    """``claude -p`` を subprocess で呼んで応答を返す."""
    if shutil.which("claude") is None:
        raise RuntimeError(
            "`claude` CLI not found on PATH. "
            "Install Claude Code (https://claude.com/claude-code) "
            "or pass --engine api."
        )
    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"claude CLI failed (exit {e.returncode}): {stderr}") from e
    return result.stdout.strip()


def _refine_via_api(
    prompt: str,
    model: str,
    max_tokens: int,
    client: Any | None,
) -> str:
    """Anthropic SDK で応答を返す."""
    if client is None:
        from anthropic import Anthropic  # 遅延 import

        client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
