---
title: SpeechRate
---

リアルタイム発話速度（モーラ/分）フィードバックアプリ。面接・原稿読み上げの練習用。
話している最中に速度メーターが見え、早口（赤）／適正（緑）／ゆっくり（青）を即座に表示する。

> このページは静的なプロジェクト紹介です。SpeechRate は Streamlit（サーバープロセス）なので
> GitHub Pages 上では動作しません。手元で `uv run main.py` を実行して使ってください。
> ライブのホスト版が欲しい場合は [Streamlit Community Cloud](https://streamlit.io/cloud) にこのリポジトリを接続できます。

## 特徴

- ハイブリッド検知：信号処理（音節核ピーク）の低遅延ライブ針＋音声認識（Vosk）の安定値とテキスト。
- 単位は「モーラ」に統一。ASR出力を pykakasi でかな化して数え、信号処理側と直接比較できる。
- ASRキャリブレーション：信号処理の系統的な過小/過大評価を ASR 比で自動補正。
- 入力レベルメーター、グラフ平滑化、セッション要約（発話速度・調音速度・適正レンジ滞在率）。
- 完全オフライン。Vosk 日本語モデルを初回のみ自動ダウンロード。

## 仕組み

| 段 | 内容 |
| --- | --- |
| 入力 | `sounddevice` でマイク捕捉（バックグラウンドスレッド）。WAV ファイル解析も可。 |
| 信号処理 | 短時間強度のピーク検出で音節核を数える（de Jong & Wempe 簡易版, numpy）。 |
| 音声認識 | Vosk `vosk-model-small-ja-0.22` → pykakasi → モーラ計数＋認識テキスト。 |
| UI | Streamlit。`st.fragment(run_every)` でライブ描画。 |

## 使い方

```sh
git clone https://github.com/P4suta/SpeechRate.git
cd SpeechRate
uv sync
uv run main.py
```

ブラウザで開いたら「録音開始」。マイクが無い場合は「WAVファイルを解析」から 16bit PCM の WAV を読み込む。

## 必要環境

- Python 3.14、[uv](https://docs.astral.sh/uv/)
- マイク（ライブ録音時）
- 初回起動時に Vosk 日本語モデル（約48MB, Apache 2.0）を自動ダウンロード。

## リンク

- [ソースコード（GitHub）](https://github.com/P4suta/SpeechRate)
- [README](https://github.com/P4suta/SpeechRate#readme)
