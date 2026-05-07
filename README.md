# mlx-whisper-pipeline

Apple Silicon Mac のローカル文字起こし + Claude による整文・要約・構造化抽出を一気通貫で行う MVP パイプライン。

機密音声（士業 / 医療 / 経営会議など）を外部送信せずに処理することを意図しています。

## できること

- mlx-whisper（Lightning Whisper MLX 互換）で音声 → 素起こしテキスト
- Claude API で素起こしを以下の形式に整形:
  - `minutes`: Markdown 形式の議事録（決定事項 / 次アクション / 質問セクション付き）
  - `summary`: 300〜500 字の plain text 要約
  - `extract`: 決定事項 / 次アクション / 質問 を JSON 構造化抽出
  - `raw`: 素起こしのみ（Claude を呼ばない）

## 動作要件

- macOS（Apple Silicon 推奨。mlx-whisper は MPS バックエンドで動作）
- Python 3.11 以上
- `ANTHROPIC_API_KEY`（`raw` 以外を使う場合）

## インストール

```bash
git clone https://github.com/TOMTOM2004/mlx-whisper-pipeline.git
cd mlx-whisper-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

開発時はテスト依存も入れる:

```bash
pip install -e ".[dev]"
pytest
```

## 使い方

### CLI

素起こしのみ:

```bash
mlx-whisper-pipeline transcribe meeting.mp3 --format raw -o transcript.txt
```

議事録 Markdown:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes -o minutes.md
```

300 字要約:

```bash
mlx-whisper-pipeline transcribe meeting.mp3 --format summary
```

決定事項 / 次アクション / 質問の JSON 抽出:

```bash
mlx-whisper-pipeline transcribe meeting.mp3 --format extract -o extracted.json
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--format` | `raw` | `raw` / `minutes` / `summary` / `extract` |
| `--whisper-model` | `mlx-community/whisper-large-v3-mlx` | HuggingFace 上の mlx-whisper モデル ID |
| `--claude-model` | `claude-sonnet-4-6` | Anthropic モデル ID |
| `--language` | （自動判定） | `ja` / `en` 等 |
| `--max-tokens` | `4096` | Claude 出力上限 |
| `-o / --output` | （stdout） | 出力ファイルパス |

### Python API

```python
from mlx_whisper_pipeline import transcribe, refine

result = transcribe("meeting.mp3", language="ja")
raw_text = result["text"]

minutes = refine(raw_text, format="minutes")
print(minutes)
```

`refine` は Anthropic クライアントを差し替え可能（テストや別 endpoint 用）:

```python
from anthropic import Anthropic
from mlx_whisper_pipeline import refine

client = Anthropic(api_key="...")
summary = refine(raw_text, format="summary", client=client)
```

## サンプル出力

[examples/sample_output.md](examples/sample_output.md) に、`--format minutes` の出力イメージを置いています。

## 設計メモ

- mlx-whisper のモデル DL は初回のみ。`~/.cache/huggingface/` にキャッシュされる。
- 機密音声では `--format raw` で素起こしを取得 → ローカルで Claude を呼ばずに別工程に渡すことも可能。
- Claude を呼ぶ場合、整文プロンプトは「推測で内容を補完しない」ルールを入れている（`mlx_whisper_pipeline/prompts.py` 参照）。

## 開発状況

MVP（v0.1）。話者分離（pyannote）・字幕 SRT 出力・GUI は未実装。

## ライセンス

MIT License — [LICENSE](LICENSE)
