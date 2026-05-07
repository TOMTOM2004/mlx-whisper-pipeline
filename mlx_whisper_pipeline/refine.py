"""素起こしテキストを整文する.

Engine は 2 種類:

- ``claude-code`` (default): subprocess で ``claude -p`` を呼ぶ。Claude Code subscription
  で動作するため API key 不要。Claude Code が PATH 上にインストールされている必要がある。
- ``api``: ``litellm`` 経由で 100+ プロバイダーに対応。``model`` パラメータで
  ``claude-sonnet-4-6`` / ``gpt-4o`` / ``gemini-2.0-flash`` / ``ollama/llama3`` 等を
  指定できる。対応する env var (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` /
  ``GEMINI_API_KEY`` / ``OLLAMA_API_BASE`` 等) は litellm が自動検出する。
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Literal

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
) -> str:
    """素起こしテキストを整文・要約・構造化する.

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
        - ``api``: ``litellm.completion`` で multi-provider 呼出
    model : str
        ``engine="api"`` のとき litellm に渡すモデル ID。
        例: ``claude-sonnet-4-6`` / ``gpt-4o`` / ``gemini-2.0-flash`` / ``ollama/llama3``。
        ``engine="claude-code"`` のときは無視される。
    max_tokens : int
        ``engine="api"`` のときの最大出力トークン数。

    Returns
    -------
    str
        モデルの出力テキスト。
    """
    if format not in PROMPT_MAP:
        raise ValueError(f"unknown format: {format!r} (expected one of {list(PROMPT_MAP)})")
    if not raw_text.strip():
        raise ValueError("raw_text is empty")

    prompt = PROMPT_MAP[format].format(text=raw_text)

    if engine == "claude-code":
        return _refine_via_claude_code(prompt)
    if engine == "api":
        return _refine_via_litellm(prompt, model=model, max_tokens=max_tokens)
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


def _refine_via_litellm(prompt: str, model: str, max_tokens: int) -> str:
    """``litellm.completion`` を呼ぶ。100+ プロバイダーに対応."""
    try:
        import litellm  # 遅延 import
    except ImportError as e:
        raise RuntimeError(
            'litellm is not installed. Install with: pip install -e ".[api]"'
        ) from e

    response = litellm.completion(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
