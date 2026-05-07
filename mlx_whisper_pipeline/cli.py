"""mlx-whisper-pipeline の CLI エントリポイント."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .refine import DEFAULT_MAX_TOKENS, DEFAULT_MODEL as DEFAULT_CLAUDE_MODEL, refine
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
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"mlx-whisper モデル ID (default: {DEFAULT_WHISPER_MODEL})",
    )
    p.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude モデル ID (default: {DEFAULT_CLAUDE_MODEL})",
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
        help=f"Claude 最大出力トークン (default: {DEFAULT_MAX_TOKENS})",
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
                model=args.claude_model,
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
