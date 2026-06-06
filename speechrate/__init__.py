"""SpeechRate — リアルタイム発話速度（モーラ/分）フィードバックのコアパッケージ。

公開モジュール:
- :mod:`speechrate.mora`     かな変換とモーラ計数（ASR経路の正規化）
- :mod:`speechrate.syllable` 信号処理による音節核ピーク検出（ライブ針）
- :mod:`speechrate.asr`      Vosk音声認識ラッパー
- :mod:`speechrate.models`   Vosk日本語モデルの自動DL/キャッシュ
- :mod:`speechrate.state`    スレッド安全な共有イベント状態
- :mod:`speechrate.metrics`  窓スライドの速度計算・分類・セッション統計
- :mod:`speechrate.audio`    マイク/WAV入力エンジン（AudioEngine）
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
