# SpeechRate

リアルタイム発話速度（モーラ/分）フィードバックアプリ。面接・原稿読み上げの練習用。
話している最中に速度メーターが見え、早口（赤）／適正（緑）／ゆっくり（青）を即座に表示する。

## 仕組み

ハイブリッド検知。単位は「モーラ」に統一している。

- 信号処理（音節核ピーク検出, numpy）: 低遅延のライブ針。
- 音声認識（Vosk + pykakasi でモーラ計数）: 安定した値と認識テキスト。
- UI: Streamlit。マイク入力は `sounddevice` をバックグラウンドスレッドで処理し、`st.fragment(run_every)` で描画する。

適正レンジの既定は 280–340 モーラ/分（アナウンサーの目安 約300 を中心）。

## 必要環境

- Python 3.12 以上、[uv](https://docs.astral.sh/uv/)
- マイク（ライブ録音時）
- 初回起動時に Vosk 日本語モデル `vosk-model-small-ja-0.22`（約48MB, Apache 2.0）を `models/` に自動ダウンロードする。

## セットアップと起動

```sh
uv sync
uv run main.py          # = uv run streamlit run app.py
```

ブラウザで開いたら「録音開始」。マイクが無い場合は「WAVファイルを解析」から 16bit PCM の WAV を読み込む。

## 開発

```sh
uv run ruff format .    # 整形
uv run ruff check .     # lint
uv run ty check         # 型チェック
uv run pytest           # テスト
```

## 構成

| パス | 役割 |
| --- | --- |
| `app.py` | Streamlit UI |
| `main.py` | 起動ランチャー |
| `speechrate/mora.py` | かな変換・モーラ計数 |
| `speechrate/syllable.py` | 音節核ピーク検出 |
| `speechrate/asr.py` | Vosk ラッパー |
| `speechrate/audio.py` | マイク/WAV 入力エンジン |
| `speechrate/state.py` | スレッド安全な共有状態 |
| `speechrate/metrics.py` | 速度計算・分類・要約 |
| `speechrate/models.py` | Vosk モデル取得 |

## 制約

- pykakasi は辞書ベースで文脈依存の読みに弱く、モーラ数は概算。
- マイクは 16kHz を試し、不可ならデバイス既定で開いて線形補間でリサンプルする。
