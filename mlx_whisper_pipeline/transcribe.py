"""mlx-whisper を呼び出して音声ファイルを文字起こしする."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"


def transcribe(
    audio_path: str | Path,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
) -> dict[str, Any]:
    """音声ファイルを mlx-whisper で文字起こしする.

    Parameters
    ----------
    audio_path : str | Path
        音声ファイルパス。mp3 / m4a / wav / flac 等。
    model : str
        HuggingFace 上の mlx-whisper モデル ID。
    language : str | None
        言語コード（"ja" / "en" 等）。None で自動判定。

    Returns
    -------
    dict
        ``mlx_whisper.transcribe`` の戻り値。``text`` / ``segments`` / ``language``
        キーを含む。
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")

    import mlx_whisper  # 遅延 import (Apple Silicon 環境のみ)

    kwargs: dict[str, Any] = {"path_or_hf_repo": model}
    if language is not None:
        kwargs["language"] = language

    return mlx_whisper.transcribe(str(path), **kwargs)
