# rvc-dataset-maker

YouTube の URL から音声を抽出し、ボーカル（BGM 除去後）を `wav` で分割して、学習用データセットを `zip` 化するアプリです。  
Colab の GPU 実行を前提にしています。

## 機能

- YouTube から音声をダウンロード（`yt-dlp`）
- `ffmpeg` でモノラル 40kHz WAV に前処理
- `demucs` でボーカル抽出（BGM 除去）
- `slicer2` で無音ベース分割
- クリップ品質フィルタ（長さ、RMS、ZCR など）
- 採用クリップを `raw/*.wav` として保存
- `metadata.json` / `README.txt` を生成
- `raw` 配下を `zip` 化
- Web UI（Gradio）で URL / 出力先入力 + 進捗ログ表示

## 必要環境

- Python 3.10 系（推奨: 3.10.20）
- [uv](https://docs.astral.sh/uv/)（推奨: 0.10.10）
- `ffmpeg`（PATH に通っていること）
- GPU ランタイム（Colab 推奨）

## セットアップ（再現性重視）

1. Python 3.10.20 を用意:

```bash
uv python install 3.10.20
```

2. `.venv` を Python 3.10.20 で作成:

```bash
uv venv --python 3.10.20
```

3. `uv.lock` に固定された依存をインストール:

```bash
uv sync --frozen
```

4. `ffmpeg` 確認:

```bash
ffmpeg -version
```

## Colab での使い方

1. ランタイムを `GPU` に変更
2. リポジトリを配置（またはアップロード）
3. 依存を同期
4. Web UI を起動

```bash
uv sync --frozen
uv run python webui.py --host 0.0.0.0 --port 7860 --share
```

`Output Path` は Colab なら例として以下を推奨:

- `/content/output`
- `/content/drive/MyDrive/rvc-output`（Google Drive 永続化）

## 使い方（CLI）

GPU 必須（デフォルト）:

```bash
uv run python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --output "/content/output"
```

CPU 実行を許可する場合（デバッグ用）:

```bash
uv run python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX" --output output --allow-cpu
```

## 使い方（Web UI）

```bash
uv run python webui.py --host 0.0.0.0 --port 7860 --share
```

UI で以下を入力して実行:

- `YouTube URL`
- `Output Path`

実行中は `進捗ログ` に `[1/6] ... [6/6]` の進行状況が表示されます。完了後に `生成ZIPのパス` が表示されます。

## 出力構成

実行後、`<output>/<dataset_name>/` が作成されます。

- `raw/*.wav`: 採用された分割クリップ
- `rejected/*.wav`: フィルタで除外されたクリップ
- `metadata.json`: クリップごとの統計情報と設定値
- `README.txt`: データセット概要
- `<dataset_name>.zip`: `raw` ディレクトリを zip 化したファイル

## 補足

- 本アプリは単体動画（`noplaylist=True`）を対象にしています。
- デフォルトは GPU 必須です。GPU が見えない場合はエラーで停止します。
- 出力データの利用時は、元コンテンツの利用規約・著作権に注意してください。
