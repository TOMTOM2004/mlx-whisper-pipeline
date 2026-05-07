from unittest.mock import patch

import pytest

from mlx_whisper_pipeline.transcribe import transcribe


def test_transcribe_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "does-not-exist.mp3")


def test_transcribe_calls_mlx_whisper_with_path(tmp_path):
    audio = tmp_path / "fake.mp3"
    audio.write_bytes(b"fake")

    fake_module = type("M", (), {})()
    fake_module.transcribe = lambda path, **kwargs: {  # type: ignore[attr-defined]
        "text": "result",
        "_path": path,
        "_kwargs": kwargs,
    }
    with patch.dict("sys.modules", {"mlx_whisper": fake_module}):
        result = transcribe(audio, model="custom/model", language="ja")

    assert result["text"] == "result"
    assert result["_path"] == str(audio)
    assert result["_kwargs"] == {"path_or_hf_repo": "custom/model", "language": "ja"}


def test_transcribe_omits_language_when_none(tmp_path):
    audio = tmp_path / "fake.mp3"
    audio.write_bytes(b"fake")

    fake_module = type("M", (), {})()
    fake_module.transcribe = lambda path, **kwargs: {  # type: ignore[attr-defined]
        "text": "x",
        "_kwargs": kwargs,
    }
    with patch.dict("sys.modules", {"mlx_whisper": fake_module}):
        result = transcribe(audio)

    assert "language" not in result["_kwargs"]
