# mlx-whisper-pipeline

Apple Silicon Mac のローカル文字起こし + Claude による整文・要約・構造化抽出を一気通貫で行う MVP パイプライン。

機密音声（士業 / 医療 / 経営会議など）を外部送信せずに処理することを意図しています。

## できること

- mlx-whisper（Lightning Whisper MLX 互換）で音声 → 素起こしテキスト
- Claude で素起こしを以下の形式に整形:
  - `minutes`: Markdown 形式の議事録（決定事項 / 次アクション / 質問セクション付き）
  - `summary`: 300〜500 字 plain text 要約
  - `extract`: 決定事項 / 次アクション / 質問 を JSON 構造化抽出
  - `raw`: 素起こしのみ（Claude を呼ばない）
- Claude 呼び出しは **2 つのエンジン** から選べる:
  - `claude-code` (default): Claude Code CLI 経由（Anthropic API key 不要、subscription で動作）
  - `api`: Anthropic SDK 経由（`ANTHROPIC_API_KEY` 必須）

## 動作要件

- macOS（Apple Silicon 推奨。mlx-whisper は MPS バックエンドで動作）
- Python 3.11 以上
- 整形を行う場合は **以下のいずれか**:
  - **Claude Code** が PATH 上にインストール済み（`claude --version` が通る）
  - もしくは `ANTHROPIC_API_KEY` を取得済み（`engine=api` 用）

## インストール

```bash
git clone https://github.com/TOMTOM2004/mlx-whisper-pipeline.git
cd mlx-whisper-pipeline
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .                # claude-code engine のみ使う場合
pip install -e ".[api]"         # api engine も使う場合（anthropic SDK 追加）
pip install -e ".[api,dev]"     # 開発時（pytest 等も入る）
pytest                           # 動作確認
```

## 使い方

### CLI

#### 素起こしのみ（Claude を呼ばない）

```bash
mlx-whisper-pipeline transcribe meeting.mp3 --format raw -o transcript.txt
```

#### 議事録 Markdown（Claude Code 経由、API key 不要）

```bash
# default engine = claude-code
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes -o minutes.md
```

#### 議事録 Markdown（Anthropic API 経由）

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-XXXX
mlx-whisper-pipeline transcribe meeting.mp3 --format minutes --engine api -o minutes.md
```

#### 300 字要約 / JSON 抽出

```bash
mlx-whisper-pipeline transcribe meeting.mp3 --format summary
mlx-whisper-pipeline transcribe meeting.mp3 --format extract -o extracted.json
```

#### 主なオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--format` | `raw` | `raw` / `minutes` / `summary` / `extract` |
| `--engine` | `claude-code` | `claude-code` / `api` |
| `--whisper-model` | `mlx-community/whisper-large-v3-mlx` | mlx-whisper モデル ID |
| `--api-model` | `claude-sonnet-4-6` | `--engine api` のときの Anthropic モデル ID |
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
print(minutes)

# Anthropic API engine（client 差し替え可能）
from anthropic import Anthropic
client = Anthropic(api_key="...")
summary = refine(raw_text, format="summary", engine="api", client=client)
```

## サンプル出力

[examples/sample_output.md](examples/sample_output.md) に、`--format minutes` の出力イメージを置いています。

## エンジンの使い分け

| 用途 | 推奨 engine | 理由 |
|---|---|---|
| 個人 PC で議事録を作る | `claude-code` | subscription 定額内で実行できコストが追加で発生しない |
| CI / 別マシン / Claude Code 未導入環境 | `api` | API key と SDK だけで動く |
| 機密音声の納品案件で顧客環境に入れる | 顧客環境次第 | Claude Code が入っていれば `claude-code`、API key を顧客側で発行できるなら `api` |

## 設計メモ

- mlx-whisper のモデル DL は初回のみ。`~/.cache/huggingface/` にキャッシュされる。
- `claude-code` engine は `claude -p "<prompt>"` を `subprocess` で呼んで stdout を受け取る形。
- 整文プロンプトは「推測で内容を補完しない」ルールを入れている（`mlx_whisper_pipeline/prompts.py` 参照）。

## 開発状況

v0.2 (Claude Code engine 追加)。話者分離（pyannote）・字幕 SRT 出力・GUI は未実装。

## ライセンス

MIT License — [LICENSE](LICENSE)
