"""Claude API で素起こしテキストを整文する."""

from __future__ import annotations

from typing import Literal

from anthropic import Anthropic

from .prompts import PROMPT_MAP

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096

Format = Literal["minutes", "summary", "extract"]


def refine(
    raw_text: str,
    format: Format = "minutes",
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Anthropic | None = None,
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
    model : str
        Anthropic モデル ID。デフォルトは Claude Sonnet 4.6。
    max_tokens : int
        最大出力トークン数。
    client : Anthropic | None
        Anthropic クライアント。None なら ``ANTHROPIC_API_KEY`` から自動生成。

    Returns
    -------
    str
        Claude の出力テキスト。
    """
    if format not in PROMPT_MAP:
        raise ValueError(f"unknown format: {format!r} (expected one of {list(PROMPT_MAP)})")

    if not raw_text.strip():
        raise ValueError("raw_text is empty")

    if client is None:
        client = Anthropic()

    prompt = PROMPT_MAP[format].format(text=raw_text)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
