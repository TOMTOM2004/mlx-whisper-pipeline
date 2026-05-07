import sys
from unittest.mock import MagicMock, patch

import pytest

from mlx_whisper_pipeline.refine import refine


def _mock_litellm(reply_text: str) -> MagicMock:
    """litellm モジュール全体を MagicMock 化したものを返す."""
    fake_litellm = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=reply_text))]
    fake_litellm.completion.return_value = fake_response
    return fake_litellm


# ---------- engine="api" (litellm) ----------


def test_refine_api_minutes_calls_litellm():
    fake_litellm = _mock_litellm("# 議事録\n\n## 決定事項\n- A")
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        result = refine("発言1。発言2。", format="minutes", engine="api")

    assert result.startswith("# 議事録")
    fake_litellm.completion.assert_called_once()
    kwargs = fake_litellm.completion.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 4096
    assert "発言1。発言2。" in kwargs["messages"][0]["content"]


def test_refine_api_summary_uses_summary_prompt():
    fake_litellm = _mock_litellm("これは要約です。")
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        result = refine("ながい素起こし...", format="summary", engine="api")

    assert result == "これは要約です。"
    prompt = fake_litellm.completion.call_args.kwargs["messages"][0]["content"]
    assert "300" in prompt


def test_refine_api_extract_uses_extract_prompt():
    fake_litellm = _mock_litellm('{"decisions": [], "next_actions": [], "questions": []}')
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        result = refine("会議の発言...", format="extract", engine="api")

    assert "decisions" in result
    prompt = fake_litellm.completion.call_args.kwargs["messages"][0]["content"]
    assert "decisions" in prompt
    assert "next_actions" in prompt


@pytest.mark.parametrize(
    "model_id",
    [
        "claude-sonnet-4-6",
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash",
        "ollama/llama3",
        "azure/my-deployment",
    ],
)
def test_refine_api_passes_arbitrary_model_to_litellm(model_id):
    fake_litellm = _mock_litellm("ok")
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        refine("text", format="summary", engine="api", model=model_id)

    assert fake_litellm.completion.call_args.kwargs["model"] == model_id


def test_refine_api_respects_custom_max_tokens():
    fake_litellm = _mock_litellm("ok")
    with patch.dict(sys.modules, {"litellm": fake_litellm}):
        refine("text", format="summary", engine="api", model="gpt-4o", max_tokens=2048)

    assert fake_litellm.completion.call_args.kwargs["max_tokens"] == 2048


def test_refine_api_missing_litellm_raises_runtime_error():
    # litellm を sys.modules から外し、import で ImportError にする
    saved = sys.modules.pop("litellm", None)
    try:
        with patch.dict(sys.modules, {"litellm": None}):
            with pytest.raises(RuntimeError, match="litellm is not installed"):
                refine("text", format="minutes", engine="api")
    finally:
        if saved is not None:
            sys.modules["litellm"] = saved


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
        refine("text", format="invalid", engine="api")  # type: ignore[arg-type]


def test_refine_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        refine("   ", format="minutes", engine="api")


def test_refine_unknown_engine_raises():
    with pytest.raises(ValueError, match="unknown engine"):
        refine("text", format="minutes", engine="invalid")  # type: ignore[arg-type]
