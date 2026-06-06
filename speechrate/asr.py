"""Vosk オフライン音声認識の薄いラッパー。

ストリーミングで int16 PCM を渡し、確定テキスト（utterance 完了時）と
部分テキスト（リアルタイム表示用）を取り出す。Vosk 日本語モデルはトークン間に
空白を挟むため、ここで除去して返す。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import vosk


def load_model(model_path: str | Path) -> vosk.Model:
    """Vosk モデルを読み込む（48MBモデルで数百ms〜）。ログは抑制する。"""
    vosk.SetLogLevel(-1)
    return vosk.Model(str(model_path))


def _text_of(payload: str, key: str) -> str:
    data: dict[str, Any] = json.loads(payload)
    return str(data.get(key, "")).replace(" ", "").strip()


class Recognizer:
    """1つの音声ストリームに対する Vosk 認識器。"""

    def __init__(self, model: vosk.Model, samplerate: int) -> None:
        self._rec = vosk.KaldiRecognizer(model, samplerate)
        self._rec.SetWords(False)

    def accept(self, pcm16: bytes) -> str | None:
        """PCM(int16, mono) を投入し、utterance が確定したら確定テキストを返す。

        まだ確定しなければ ``None``。確定テキストは句読点・空白を除いた読み上げ列。
        """
        if self._rec.AcceptWaveform(pcm16):
            return _text_of(self._rec.Result(), "text") or None
        return None

    def partial(self) -> str:
        """現在の部分認識テキスト（リアルタイム表示用、確定前）。"""
        return _text_of(self._rec.PartialResult(), "partial")

    def flush(self) -> str:
        """ストリーム終端で残りの確定テキストを取り出す。"""
        return _text_of(self._rec.FinalResult(), "text")
