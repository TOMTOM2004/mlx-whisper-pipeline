"""mlx-whisper-pipeline の CLI エントリポイント."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .refine import (
    DEFAULT_API_MODEL,
    DEFAULT_ENGINE,
    DEFAULT_MAX_TOKENS,
    refine,
)
from .transcribe import DEFAULT_MODEL as DEFAULT_WHISPER_MODEL, transcribe


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-whisper-pipeline",
        description="Apple Silicon ローカル文字起こし + Claude 整文パイプライン",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "transcribe",
        help="音声ファイルを文字起こし、必要なら Claude で整文する",
    )
    p.add_argument("audio", type=Path, help="音声ファイルパス（mp3/m4a/wav/flac 等）")
    p.add_argument(
        "--format",
        choices=["raw", "minutes", "summary", "extract"],
        default="raw",
        help="出力フォーマット (default: raw = 素起こしのみ)",
    )
    p.add_argument(
        "--engine",
        choices=["claude-code", "api"],
        default=DEFAULT_ENGINE,
        help=(
            f"整文エンジン (default: {DEFAULT_ENGINE})。"
            '"claude-code" は claude CLI 経由（subscription、API key 不要）。'
            '"api" は Anthropic SDK 経由（ANTHROPIC_API_KEY 必須）'
        ),
    )
    p.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"mlx-whisper モデル ID (default: {DEFAULT_WHISPER_MODEL})",
    )
    p.add_argument(
        "--api-model",
        default=DEFAULT_API_MODEL,
        help=(
            f"Anthropic モデル ID (default: {DEFAULT_API_MODEL})。"
            "--engine api のときのみ有効"
        ),
    )
    p.add_argument(
        "--language",
        default=None,
        help='言語コード（例: "ja", "en"）。未指定で自動判定',
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            f"Claude 最大出力トークン (default: {DEFAULT_MAX_TOKENS})。"
            "--engine api のときのみ有効"
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="出力先ファイル（未指定で stdout）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "transcribe":
        result = transcribe(
            args.audio,
            model=args.whisper_model,
            language=args.language,
        )
        raw_text = result["text"]

        if args.format == "raw":
            output = raw_text
        else:
            output = refine(
                raw_text,
                format=args.format,
                engine=args.engine,
                model=args.api_model,
                max_tokens=args.max_tokens,
            )

        if args.output:
            args.output.write_text(output, encoding="utf-8")
            print(f"wrote: {args.output}", file=sys.stderr)
        else:
            print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
