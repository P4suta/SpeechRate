"""スレッド安全な共有イベント状態。

オーディオのワーカースレッドが書き込み（音節ピーク・ASR確定テキスト・部分テキスト・
クロック）、Streamlit のフラグメント（UIスレッド）が :meth:`snapshot` で一貫した
コピーを読む。時刻はすべてオーディオのストリームクロック（サンプル数/サンプルレート）で、
ライブでもファイル再生でも一貫する。

直近の窓（既定120秒）を超えた音節・履歴は刈り取りつつ、セッション要約用の累積値
（総音節数・ASR総モーラ数）は別に保持する。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AsrUtterance:
    """確定した1発話。"""

    time: float  # 確定時のストリーム時刻（秒）
    text: str
    morae: int


@dataclass(frozen=True)
class Snapshot:
    """UI が一度に読むための状態コピー。"""

    stream_time: float
    voiced_seconds: float
    syllable_times: tuple[float, ...]
    total_syllables: int
    utterances: tuple[AsrUtterance, ...]
    total_asr_morae: int
    partial: str
    rate_history: tuple[tuple[float, float], ...]
    level: float = 0.0  # 直近ブロックの入力レベル（ピーク振幅 0..1）


class SharedState:
    """ワーカー（書き込み）と UI（読み取り）をつなぐロック付きの状態。"""

    def __init__(self, history_sec: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._history_sec = history_sec
        self._syllable_times: deque[float] = deque()
        self._utterances: deque[AsrUtterance] = deque()
        self._rate_history: deque[tuple[float, float]] = deque()
        self._partial = ""
        self._stream_time = 0.0
        self._voiced_seconds = 0.0
        self._total_syllables = 0
        self._total_asr_morae = 0
        self._level = 0.0

    def reset(self) -> None:
        with self._lock:
            self._syllable_times.clear()
            self._utterances.clear()
            self._rate_history.clear()
            self._partial = ""
            self._stream_time = 0.0
            self._voiced_seconds = 0.0
            self._total_syllables = 0
            self._total_asr_morae = 0
            self._level = 0.0

    def add_syllables(self, times: list[float]) -> None:
        if not times:
            return
        with self._lock:
            self._syllable_times.extend(times)
            self._total_syllables += len(times)

    def add_utterance(self, utterance: AsrUtterance) -> None:
        with self._lock:
            self._utterances.append(utterance)
            self._total_asr_morae += utterance.morae

    def set_partial(self, text: str) -> None:
        with self._lock:
            self._partial = text

    def set_level(self, level: float) -> None:
        with self._lock:
            self._level = level

    def update_clock(self, stream_time: float, voiced_seconds: float) -> None:
        with self._lock:
            self._stream_time = stream_time
            self._voiced_seconds = voiced_seconds
            self._prune_locked()

    def push_rate(self, stream_time: float, rate: float) -> None:
        with self._lock:
            self._rate_history.append((stream_time, rate))

    def _prune_locked(self) -> None:
        cutoff = self._stream_time - self._history_sec
        while self._syllable_times and self._syllable_times[0] < cutoff:
            self._syllable_times.popleft()
        while self._rate_history and self._rate_history[0][0] < cutoff:
            self._rate_history.popleft()
        # トランスクリプトは直近の発話のみ画面表示用に残す（窓よりやや多め）。
        while len(self._utterances) > 200:
            self._utterances.popleft()

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                stream_time=self._stream_time,
                voiced_seconds=self._voiced_seconds,
                syllable_times=tuple(self._syllable_times),
                total_syllables=self._total_syllables,
                utterances=tuple(self._utterances),
                total_asr_morae=self._total_asr_morae,
                partial=self._partial,
                rate_history=tuple(self._rate_history),
                level=self._level,
            )
