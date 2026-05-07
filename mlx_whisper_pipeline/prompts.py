"""Claude に渡す整文プロンプトテンプレート."""

MINUTES_PROMPT = """以下の素起こしテキストを議事録形式に整形してください。

## 出力ルール
- Markdown ヘッダ（##, ###）で章立てする
- 言い淀み・繰り返しを除去し、要点ベースで整理する
- 発言者ラベルが原文にあれば保持する
- 末尾に以下のセクションを設ける:
  - ## 決定事項
  - ## 次アクション
  - ## 質問・保留事項
- 推測で内容を補完しない。原文に無い情報は加えない。

## 素起こし

{text}
"""

SUMMARY_PROMPT = """以下の素起こしテキストを 300〜500 字で要約してください。

## 出力ルール
- 一段落の plain text（Markdown 不要）
- 主要なトピック・結論・残課題のみを含める
- 推測で内容を補完しない

## 素起こし

{text}
"""

EXTRACT_PROMPT = """以下の素起こしテキストから、決定事項・次アクション・質問を JSON で抽出してください。

## 出力スキーマ
```json
{{
  "decisions": [{{"summary": "...", "owner": "..." or null}}],
  "next_actions": [{{"summary": "...", "owner": "..." or null, "due": "..." or null}}],
  "questions": [{{"summary": "...", "raised_by": "..." or null}}]
}}
```

## 出力ルール
- 必ず上記の JSON のみを返す（前後のテキスト不要）
- 原文に無い情報は推測で補完しない
- owner / due / raised_by は明示が無ければ null

## 素起こし

{text}
"""

PROMPT_MAP = {
    "minutes": MINUTES_PROMPT,
    "summary": SUMMARY_PROMPT,
    "extract": EXTRACT_PROMPT,
}
