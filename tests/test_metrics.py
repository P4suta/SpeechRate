"""窓スライド速度・分類・セッション要約の検証。"""

from __future__ import annotations

from speechrate import metrics
from speechrate.state import AsrUtterance, Snapshot


def test_live_rate_sliding_window() -> None:
    # 0.2秒間隔のピーク = 5音節/秒 = 300モーラ/分。
    times = tuple(i * 0.2 for i in range(50))
    assert metrics.live_rate(times, now=10.0, window=4.0) == 300.0


def test_live_rate_short_history_uses_elapsed() -> None:
    # 経過が窓より短いときは実経過で割って過小評価を避ける。
    times = (0.2, 0.4, 0.6, 0.8, 1.0)  # 2秒時点で5モーラ
    assert metrics.live_rate(times, now=2.0, window=4.0) == 150.0


def test_calibration_factor() -> None:
    # 十分なサンプルがあれば ASRモーラ ÷ 信号ピーク。
    assert metrics.calibration_factor(20, 30) == 1.5
    # サンプル不足は無補正。
    assert metrics.calibration_factor(5, 30) == 1.0
    assert metrics.calibration_factor(20, 3) == 1.0
    # 極端な比は [0.5, 2.0] に制限。
    assert metrics.calibration_factor(20, 100) == 2.0
    assert metrics.calibration_factor(100, 10) == 0.5


def test_summarize_applies_calibration() -> None:
    snap = Snapshot(
        stream_time=10.0,
        voiced_seconds=10.0,
        syllable_times=(),
        total_syllables=20,  # 信号は過小評価
        utterances=(),
        total_asr_morae=30,  # ASR が正、補正係数 1.5
        partial="",
        rate_history=(),
    )
    s = metrics.summarize(snap, low=280, high=340)
    # 補正後モーラ = 20*1.5 = 30 → 30/10*60 = 180
    assert round(s.speech_rate) == 180


def test_classify_boundaries() -> None:
    assert metrics.classify(0, 280, 340).label == "待機中"
    assert metrics.classify(200, 280, 340).label == "ゆっくり"
    assert metrics.classify(300, 280, 340).label == "適正"
    assert metrics.classify(400, 280, 340).label == "早口"


def test_summarize() -> None:
    snap = Snapshot(
        stream_time=10.0,
        voiced_seconds=8.0,
        syllable_times=tuple(i * 0.2 for i in range(50)),
        total_syllables=50,
        utterances=(AsrUtterance(5.0, "てすと", 3),),
        total_asr_morae=3,
        partial="",
        rate_history=((1.0, 300.0), (2.0, 290.0), (3.0, 500.0)),
    )
    s = metrics.summarize(snap, low=280, high=340)
    assert round(s.speech_rate) == 300  # 50/10*60
    assert round(s.articulation_rate) == 375  # 50/8*60
    assert s.pause_seconds == 2.0
    assert s.peak_rate == 500.0
    assert 0.0 < s.pct_in_range < 1.0  # 300,290 が範囲内 / 500 が範囲外
