"""音節核検出の検証。

合成した振幅変調音（既知の音節数）を流し、検出数・無音時0・
ブロックサイズ非依存（ストリーミングの取りこぼし/二重計数なし）を確認する。
"""

from __future__ import annotations

import numpy as np

from speechrate.syllable import SyllableDetector

SR = 16000


def _synth(n_syllables: int, duration: float) -> np.ndarray:
    """duration 秒に n_syllables 個の鋭い母音核を持つモノラル信号。"""
    t = np.arange(int(SR * duration)) / SR
    rate = n_syllables / duration
    env = (0.5 * (1.0 + np.sin(2 * np.pi * rate * t - np.pi / 2))) ** 3
    sig = env * np.sin(2 * np.pi * 200 * t)
    return (sig * 0.3).astype(np.float32)


def _run(signal: np.ndarray, block: int) -> int:
    det = SyllableDetector(SR)
    peaks: list[float] = []
    for i in range(0, len(signal), block):
        peaks += det.process(signal[i : i + block])
    return len(peaks)


def test_counts_known_syllables() -> None:
    signal = _synth(10, 2.0)  # 5音節/秒
    assert 8 <= _run(signal, 1600) <= 12


def test_silence_yields_no_peaks() -> None:
    signal = np.zeros(SR, dtype=np.float32)
    assert _run(signal, 1600) == 0


def test_block_size_invariant() -> None:
    """ブロック分割が変わっても結果は同一（carry とスライド窓が連続している証拠）。"""
    signal = _synth(12, 2.4)
    # 160サンプル境界に揃わないサイズも混ぜて carry 経路を踏ませる。
    counts = {_run(signal, block) for block in (777, 1000, 1600, len(signal))}
    assert len(counts) == 1


def test_voiced_seconds_tracks_phonation() -> None:
    det = SyllableDetector(SR)
    det.process(_synth(8, 1.6))
    # 発声時間は総時間以下で、無音だらけでなければ正の値になる。
    assert 0.0 < det.voiced_seconds <= det.stream_time
