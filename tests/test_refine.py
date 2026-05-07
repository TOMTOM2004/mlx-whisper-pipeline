from unittest.mock import MagicMock, patch

import pytest

from mlx_whisper_pipeline.refine import refine


def _mock_api_client(reply_text: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=reply_text)]
    client.messages.create.return_value = response
    return client


# ---------- engine="api" ----------


def test_refine_api_minutes_calls_claude_with_prompt():
    client = _mock_api_client("# 議事録\n\n## 決定事項\n- A")
    result = refine("発言1。発言2。", format="minutes", engine="api", client=client)

    assert result.startswith("# 議事録")
    client.messages.create.assert_called_once()
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert "発言1。発言2。" in kwargs["messages"][0]["content"]


def test_refine_api_summary_uses_summary_prompt():
    client = _mock_api_client("これは要約です。")
    result = refine("ながい素起こし...", format="summary", engine="api", client=client)

    assert result == "これは要約です。"
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "300" in prompt


def test_refine_api_extract_uses_extract_prompt():
    client = _mock_api_client('{"decisions": [], "next_actions": [], "questions": []}')
    result = refine("会議の発言...", format="extract", engine="api", client=client)

    assert "decisions" in result
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "decisions" in prompt
    assert "next_actions" in prompt


def test_refine_api_respects_custom_model_and_max_tokens():
    client = _mock_api_client("ok")
    refine(
        "text",
        format="summary",
        engine="api",
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        client=client,
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 2048


# ---------- engine="claude-code" ----------


def test_refine_claude_code_invokes_claude_cli():
    fake_run_result = MagicMock()
    fake_run_result.stdout = "  # 議事録\n\n本文  "
    fake_run_result.returncode = 0
    with (
        patch("mlx_whisper_pipeline.refine.shutil.which", return_value="/usr/local/bin/claude"),
        patch("mlx_whisper_pipeline.refine.subprocess.run", return_value=fake_run_result) as mock_run,
    ):
        result = refine("発言。", format="minutes", engine="claude-code")

    assert result == "# 議事録\n\n本文"
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "claude"
    assert cmd[1] == "-p"
    assert "発言。" in cmd[2]
    assert kwargs["check"] is True
    assert kwargs["text"] is True


def test_refine_claude_code_default_engine_is_claude_code():
    fake_run_result = MagicMock()
    fake_run_result.stdout = "ok"
    with (
        patch("mlx_whisper_pipeline.refine.shutil.which", return_value="/usr/local/bin/claude"),
        patch("mlx_whisper_pipeline.refine.subprocess.run", return_value=fake_run_result) as mock_run,
    ):
        refine("text", format="summary")

    mock_run.assert_called_once()


def test_refine_claude_code_missing_cli_raises_runtime_error():
    with patch("mlx_whisper_pipeline.refine.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="claude.*not found"):
            refine("text", format="minutes", engine="claude-code")


def test_refine_claude_code_propagates_called_process_error():
    import subprocess as sp

    err = sp.CalledProcessError(returncode=1, cmd=["claude"], stderr="bad request")
    with (
        patch("mlx_whisper_pipeline.refine.shutil.which", return_value="/usr/local/bin/claude"),
        patch("mlx_whisper_pipeline.refine.subprocess.run", side_effect=err),
    ):
        with pytest.raises(RuntimeError, match="claude CLI failed.*bad request"):
            refine("text", format="minutes", engine="claude-code")


# ---------- validation ----------


def test_refine_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        refine("text", format="invalid", engine="api", client=_mock_api_client(""))  # type: ignore[arg-type]


def test_refine_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        refine("   ", format="minutes", engine="api", client=_mock_api_client(""))


def test_refine_unknown_engine_raises():
    with pytest.raises(ValueError, match="unknown engine"):
        refine("text", format="minutes", engine="invalid")  # type: ignore[arg-type]
