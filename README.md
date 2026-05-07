# mlx-whisper-pipeline

Apple Silicon Mac のローカル文字起こし + LLM による整文・要約・構造化抽出を一気通貫で行う MVP パイプライン。

機密音声（士業 / 医療 / 経営会議など）を外部送信せずに処理することを意図しています。

## できること

- mlx-whisper（Lightning Whisper MLX 互換）で音声 → 素起こしテキスト
- LLM で素起こしを以下の形式に整形:
  - `minutes`: Markdown 形式の議事録（決定事項 / 次アクション / 質問セクション付き）
  - `summary`: 300〜500 字 plain text 要約
  - `extract`: 決定事項 / 次アクション / 質問 を JSON 構造化抽出
  - `raw`: 素起こしのみ（LLM を呼ばない）
- LLM 呼び出しは **2 つのエンジン** から選べる:
  - `claude-code` (default): Claude Code CLI 経由（API key 不要、Claude Code subscription で動作）
  - `api`: [litellm](https://github.com/BerriAI/litellm) 経由で **100+ プロバイダー対応**（Anthropic / OpenAI / Gemini / Azure / Bedrock / Cohere / Ollama 等）

## 動作要件

- macOS（Apple Silicon 推奨。mlx-whisper は MPS バックエンドで動作）
- Python 3.11 以上
- 整形を行う場合は **以下のいずれか**:
  - **Claude Code** が PATH 上にインストール済み（`claude --version` が通る）
  - もしくは `litellm` + 対応 provider の env var（`engine=api` 用）

## インストール

```bash
git clone https://github.com/TOMTOM2004/mlx-whisper-pipeline.git
cd mlx-whisper-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .                # claude-code engine のみ使う場合
pip install -e ".[api]"         # api engine も使う場合（litellm 追加）
pip install -e ".[api,dev]"     # 開発時（pytest 等も入る）
pytest                           # 動作確認
```

## 使い方

### CLI

#### 素起こしのみ（LLM を呼ばない）

```bash
mlx-whisper-pipeline transcribe meeting.mp3 --format raw -o transcript.txt
```

#### 議事録 Markdown（Claude Code 経由、API key 不要）

```bash
# default engine = claude-code
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes -o minutes.md
```

#### 議事録 Markdown（litellm 経由、provider 自由選択）

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes \
    --engine api --api-model claude-sonnet-4-6 -o minutes.md

# OpenAI
export OPENAI_API_KEY=sk-...
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes \
    --engine api --api-model gpt-4o -o minutes.md

# Google Gemini
export GEMINI_API_KEY=...
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes \
    --engine api --api-model gemini-2.0-flash -o minutes.md

# Ollama (完全ローカル LLM)
ollama pull llama3
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes \
    --engine api --api-model ollama/llama3 -o minutes.md
```

#### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--format` | `raw` | `raw` / `minutes` / `summary` / `extract` |
| `--engine` | `claude-code` | `claude-code` / `api` |
| `--whisper-model` | `mlx-community/whisper-large-v3-mlx` | mlx-whisper モデル ID |
| `--api-model` | `claude-sonnet-4-6` | `--engine api` のときの litellm モデル ID |
| `--language` | （自動判定） | `ja` / `en` 等 |
| `--max-tokens` | `4096` | `--engine api` のときの最大出力トークン |
| `-o / --output` | （stdout） | 出力ファイルパス |

### Python API

```python
from mlx_whisper_pipeline import transcribe, refine

result = transcribe("meeting.mp3", language="ja")
raw_text = result["text"]

# Claude Code engine（default）
minutes = refine(raw_text, format="minutes")

# litellm engine（任意 provider）
summary = refine(raw_text, format="summary", engine="api", model="gpt-4o")
extracted_json = refine(raw_text, format="extract", engine="api", model="gemini-2.0-flash")
```

## 対応 provider 例（`--engine api`）

litellm の対応モデル ID をそのまま `--api-model` に指定できます。代表例:

| Provider | モデル例 | 必要な env var |
|---|---|---|
| Anthropic | `claude-sonnet-4-6` / `claude-opus-4-7` / `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| OpenAI | `gpt-4o` / `gpt-4o-mini` / `o1` | `OPENAI_API_KEY` |
| Google | `gemini-2.0-flash` / `gemini-1.5-pro` | `GEMINI_API_KEY` |
| Azure OpenAI | `azure/<deployment-name>` | `AZURE_API_KEY` / `AZURE_API_BASE` |
| AWS Bedrock | `bedrock/anthropic.claude-3-5-sonnet-...` | AWS credentials (boto3 互換) |
| Cohere | `command-r-plus` | `COHERE_API_KEY` |
| Ollama (local) | `ollama/llama3` / `ollama/qwen2.5` | `OLLAMA_API_BASE` (default: `http://localhost:11434`) |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |

完全な対応リストは [litellm Providers ドキュメント](https://docs.litellm.ai/docs/providers) を参照。

## サンプル出力

[examples/sample_output.md](examples/sample_output.md) に、`--format minutes` の出力イメージを置いています。

## エンジンの使い分け

| 用途 | 推奨 engine | 理由 |
|---|---|---|
| 個人 PC で議事録を作る | `claude-code` | subscription 定額内で実行できコストが追加で発生しない |
| 顧客指定が OpenAI / Gemini 等 | `api` (該当 provider) | litellm でモデル ID を変えるだけで対応 |
| CI / 別マシン / Claude Code 未導入環境 | `api` (お好みの provider) | env var だけで動く |
| 完全オフライン / 機密で API も避けたい | `api` (Ollama) | ローカル LLM のみで完結 |

## 設計メモ

- mlx-whisper のモデル DL は初回のみ。`~/.cache/huggingface/` にキャッシュされる。
- `claude-code` engine は `claude -p "<prompt>"` を `subprocess` で呼んで stdout を受け取る形。
- `api` engine は `litellm.completion()` を呼ぶ。litellm が provider ごとの env var を自動検出する。
- 整文プロンプトは「推測で内容を補完しない」ルールを入れている（`mlx_whisper_pipeline/prompts.py` 参照）。

## 開発状況

v0.3 (litellm 導入で multi-provider 化)。話者分離（pyannote）・字幕 SRT 出力・GUI は未実装。

## ライセンス

MIT License — [LICENSE](LICENSE)
