"""窓スライドの速度計算・分類・セッション要約。

すべて「モーラ/分」を単位とする。信号処理（音節核）由来とASR（モーラ計数）由来の
2系統を同じ単位で扱えるため、互いに比較でき、アナウンサー基準（≒300モーラ/分）とも
噛み合う。
"""

from __future__ import annotations

from dataclasses import dataclass

from speechrate.state import AsrUtterance, Snapshot

# 既定の適正レンジ（モーラ/分）。アナウンサーの聞き取りやすい速さ ~300 を中心に取る。
DEFAULT_TARGET_LOW = 280.0
DEFAULT_TARGET_HIGH = 340.0

# 窓長（秒）。ライブ針は短く即応、ASR窓は長めに安定。
DEFAULT_LIVE_WINDOW = 4.0
DEFAULT_ASR_WINDOW = 10.0

# ASRキャリブレーション: 信号処理の音節核検出は話者・声質で系統的に過小/過大評価し得る。
# 累積の ASRモーラ数 ÷ 信号ピーク数 を補正係数として針に掛け、絶対値を ASR に寄せる。
# 十分なサンプルが貯まるまでは 1.0（無補正）、極端な値は [0.5, 2.0] に制限する。
CALIB_MIN_SYLLABLES = 10
CALIB_MIN_MORAE = 5
CALIB_FACTOR_RANGE = (0.5, 2.0)


def calibration_factor(total_syllables: int, total_asr_morae: int) -> float:
    """信号処理の針を ASR に寄せる補正係数。"""
    if total_syllables < CALIB_MIN_SYLLABLES or total_asr_morae < CALIB_MIN_MORAE:
        return 1.0
    low, high = CALIB_FACTOR_RANGE
    return min(high, max(low, total_asr_morae / total_syllables))


@dataclass(frozen=True)
class Classification:
    """速度の分類結果（ラベルとUI表示色）。"""

    label: str
    color: str  # CSS色


def _rate(count: float, now: float, window: float) -> float:
    """直近 ``window`` 秒の件数から「件/分」を求める。

    発話開始直後（``now < window``）は実経過時間で割って過小評価を避ける。
    """
    effective = min(window, now)
    if effective <= 0:
        return 0.0
    return count * 60.0 / effective


def live_rate(
    syllable_times: tuple[float, ...],
    now: float,
    window: float = DEFAULT_LIVE_WINDOW,
) -> float:
    """信号処理ベースの瞬間発話速度（モーラ/分）。"""
    start = now - window
    count = sum(1 for t in syllable_times if t >= start)
    return _rate(count, now, window)


def asr_rate(
    utterances: tuple[AsrUtterance, ...],
    now: float,
    window: float = DEFAULT_ASR_WINDOW,
) -> float:
    """ASRベースの安定発話速度（モーラ/分）。"""
    start = now - window
    morae = sum(u.morae for u in utterances if u.time >= start)
    return _rate(morae, now, window)


def classify(rate: float, low: float, high: float) -> Classification:
    """速度を 待機中/ゆっくり/適正/早口 に分類する。"""
    if rate <= 0:
        return Classification("待機中", "#9aa0a6")
    if rate < low:
        return Classification("ゆっくり", "#1a73e8")
    if rate > high:
        return Classification("早口", "#d93025")
    return Classification("適正", "#1e8e3e")


@dataclass(frozen=True)
class SessionSummary:
    """停止時に表示するセッション統計。"""

    duration: float  # 総時間（秒）
    voiced_seconds: float  # 発声時間（秒）
    pause_seconds: float  # 無音/ポーズ時間（秒）
    total_syllables: int  # 信号処理で数えた総モーラ
    total_asr_morae: int  # ASRで数えた総モーラ
    speech_rate: float  # 発話速度: 総モーラ/総時間（ポーズ込み）
    articulation_rate: float  # 調音速度: 総モーラ/発声時間（ポーズ除外）
    asr_speech_rate: float  # ASRモーラ/総時間
    pct_in_range: float  # 瞬間速度が適正レンジ内だった割合(0-1)
    peak_rate: float  # 瞬間最大速度
    min_rate: float  # 発話中の瞬間最小速度


def summarize(snap: Snapshot, low: float, high: float) -> SessionSummary:
    """スナップショットからセッション要約を作る。"""
    duration = snap.stream_time
    voiced = min(snap.voiced_seconds, duration)
    pause = max(0.0, duration - voiced)

    def per_min(count: float, denom: float) -> float:
        return count * 60.0 / denom if denom > 0 else 0.0

    # 信号処理ベースの速度は ASR で補正してから提示する（針・チャートと整合）。
    factor = calibration_factor(snap.total_syllables, snap.total_asr_morae)
    effective_morae = snap.total_syllables * factor

    rates = [r for _, r in snap.rate_history]
    active = [r for r in rates if r > 0]
    in_range = [r for r in active if low <= r <= high]
    pct = len(in_range) / len(active) if active else 0.0

    return SessionSummary(
        duration=duration,
        voiced_seconds=voiced,
        pause_seconds=pause,
        total_syllables=snap.total_syllables,
        total_asr_morae=snap.total_asr_morae,
        speech_rate=per_min(effective_morae, duration),
        articulation_rate=per_min(effective_morae, voiced),
        asr_speech_rate=per_min(snap.total_asr_morae, duration),
        pct_in_range=pct,
        peak_rate=max(active) if active else 0.0,
        min_rate=min(active) if active else 0.0,
    )
