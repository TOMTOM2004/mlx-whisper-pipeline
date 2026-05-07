from unittest.mock import MagicMock

import pytest

from mlx_whisper_pipeline.refine import refine


def _mock_client(reply_text: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=reply_text)]
    client.messages.create.return_value = response
    return client


def test_refine_minutes_calls_claude_with_prompt():
    client = _mock_client("# 議事録\n\n## 決定事項\n- A")
    result = refine("発言1。発言2。", format="minutes", client=client)

    assert result.startswith("# 議事録")
    client.messages.create.assert_called_once()
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert "発言1。発言2。" in kwargs["messages"][0]["content"]


def test_refine_summary_uses_summary_prompt():
    client = _mock_client("これは要約です。")
    result = refine("ながい素起こし...", format="summary", client=client)

    assert result == "これは要約です。"
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "300" in prompt


def test_refine_extract_uses_extract_prompt():
    client = _mock_client('{"decisions": [], "next_actions": [], "questions": []}')
    result = refine("会議の発言...", format="extract", client=client)

    assert "decisions" in result
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "decisions" in prompt
    assert "next_actions" in prompt


def test_refine_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        refine("text", format="invalid", client=_mock_client(""))  # type: ignore[arg-type]


def test_refine_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        refine("   ", format="minutes", client=_mock_client(""))


def test_refine_respects_custom_model_and_max_tokens():
    client = _mock_client("ok")
    refine(
        "text",
        format="summary",
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        client=client,
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 2048
