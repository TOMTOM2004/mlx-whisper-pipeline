from pathlib import Path
from unittest.mock import patch

import pytest

from mlx_whisper_pipeline.cli import _build_parser, main


def test_parser_requires_subcommand(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_transcribe_defaults():
    parser = _build_parser()
    args = parser.parse_args(["transcribe", "audio.mp3"])
    assert args.cmd == "transcribe"
    assert args.audio == Path("audio.mp3")
    assert args.format == "raw"
    assert args.language is None


def test_parser_transcribe_with_format_and_output():
    parser = _build_parser()
    args = parser.parse_args(
        ["transcribe", "audio.mp3", "--format", "minutes", "-o", "out.md"]
    )
    assert args.format == "minutes"
    assert args.output == Path("out.md")


def test_main_raw_calls_transcribe_only(capsys):
    with (
        patch("mlx_whisper_pipeline.cli.transcribe", return_value={"text": "hello"}) as mock_t,
        patch("mlx_whisper_pipeline.cli.refine") as mock_r,
    ):
        rc = main(["transcribe", "audio.mp3", "--format", "raw"])

    assert rc == 0
    mock_t.assert_called_once()
    mock_r.assert_not_called()
    out = capsys.readouterr().out
    assert "hello" in out


def test_main_minutes_calls_refine(capsys):
    with (
        patch("mlx_whisper_pipeline.cli.transcribe", return_value={"text": "raw text"}),
        patch("mlx_whisper_pipeline.cli.refine", return_value="# Minutes") as mock_r,
    ):
        rc = main(["transcribe", "audio.mp3", "--format", "minutes"])

    assert rc == 0
    mock_r.assert_called_once_with(
        "raw text",
        format="minutes",
        model="claude-sonnet-4-6",
        max_tokens=4096,
    )
    assert "# Minutes" in capsys.readouterr().out


def test_main_writes_to_output_file(tmp_path, capsys):
    out_file = tmp_path / "out.md"
    with (
        patch("mlx_whisper_pipeline.cli.transcribe", return_value={"text": "raw"}),
        patch("mlx_whisper_pipeline.cli.refine", return_value="result"),
    ):
        main(["transcribe", "audio.mp3", "--format", "summary", "-o", str(out_file)])

    assert out_file.read_text(encoding="utf-8") == "result"
