from mlx_whisper_pipeline.prompts import EXTRACT_PROMPT, MINUTES_PROMPT, PROMPT_MAP, SUMMARY_PROMPT


def test_prompt_map_keys():
    assert set(PROMPT_MAP.keys()) == {"minutes", "summary", "extract"}


def test_prompts_have_text_placeholder():
    for name, template in PROMPT_MAP.items():
        formatted = template.format(text="サンプル")
        assert "サンプル" in formatted, f"{name} missing {{text}} placeholder"


def test_minutes_prompt_includes_required_sections():
    assert "決定事項" in MINUTES_PROMPT
    assert "次アクション" in MINUTES_PROMPT
    assert "質問" in MINUTES_PROMPT


def test_summary_prompt_specifies_length():
    assert "300" in SUMMARY_PROMPT


def test_extract_prompt_specifies_json_keys():
    assert "decisions" in EXTRACT_PROMPT
    assert "next_actions" in EXTRACT_PROMPT
    assert "questions" in EXTRACT_PROMPT
