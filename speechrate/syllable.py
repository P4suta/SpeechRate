"""信号処理による音節核（モーラ）検出 — 低遅延の「ライブ針」。

de Jong & Wempe (2009) の Praat スクリプト "Syllable Nuclei" の考え方を、
numpy のみで軽量・ストリーミング向けに簡略化したもの。

各フレームの強度（dB）を求め、以下を満たす局所最大を音節核とみなす:
- 無音閾値（走行最大強度から ``drop_db`` 下／絶対フロア）以上。
- 直前に受理したピークから ``dip_db`` 以上の谷（prominence）を挟む。
- 直前のピークから ``min_dist_sec`` 以上離れている（最大 ~8音節/秒）。

フォルマント/有声性の判定は省略しているため厳密な音節検出ではないが、
発話速度（数秒窓のモーラ/分）の推定には十分。ピーク時刻は「ストリーム開始からの秒」で、
オーディオのサンプルクロックに基づくため、ライブでもファイル再生でも一貫する。
"""

from __future__ import annotations

import math

import numpy as np


class SyllableDetector:
    """ストリーミング音声から音節核ピークの時刻列を検出する。

    :meth:`process` にモノラル float32（``[-1, 1]``）のブロックを順次渡すと、
    そのブロックで確定したピークの時刻（ストリーム開始からの秒）のリストを返す。
    ブロック境界をまたいでも3フレームのスライド窓で状態を保持するため、
    取りこぼし・二重計数を起こさない。
    """

    def __init__(
        self,
        samplerate: int,
        *,
        frame_ms: float = 10.0,
        drop_db: float = 25.0,
        floor_db: float = -55.0,
        dip_db: float = 2.0,
        min_dist_sec: float = 0.12,
        max_decay_db_per_sec: float = 4.0,
        voice_drop_db: float = 35.0,
    ) -> None:
        self.samplerate = samplerate
        self._frame_len = max(1, round(samplerate * frame_ms / 1000.0))
        self._hop_sec = self._frame_len / samplerate
        self._drop_db = drop_db
        self._floor_db = floor_db
        self._dip_db = dip_db
        self._min_dist_sec = min_dist_sec
        self._max_decay_per_frame = max_decay_db_per_sec * self._hop_sec
        # 発声（phonation）判定はピーク用より広めの閾値で、子音や減衰部も拾う。
        self._voice_drop_db = voice_drop_db
        self.reset()

    def reset(self) -> None:
        """セッション境界で内部状態を初期化する。"""
        self._carry = np.empty(0, dtype=np.float32)
        self._frame_counter = 0
        # 3フレームのスライド窓: 直前2フレーム (global_index, db)
        self._f0: tuple[int, float] | None = None
        self._f1: tuple[int, float] | None = None
        self._running_max_db = self._floor_db
        self._min_since_peak = self._floor_db
        self._last_peak_idx = -(10**9)
        self._voiced_frames = 0

    @property
    def stream_time(self) -> float:
        """これまでに処理した音声の長さ（秒）。"""
        return self._frame_counter * self._hop_sec

    @property
    def voiced_seconds(self) -> float:
        """発声していた時間の累計（秒）。調音速度の分母に使う。"""
        return self._voiced_frames * self._hop_sec

    def process(self, samples: np.ndarray) -> list[float]:
        """1ブロック分の音声を処理し、確定したピーク時刻（秒）のリストを返す。"""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        buf = np.concatenate([self._carry, samples]) if self._carry.size else samples
        n_frames = buf.size // self._frame_len
        peaks: list[float] = []
        fl = self._frame_len
        for i in range(n_frames):
            frame = buf[i * fl : (i + 1) * fl]
            rms = math.sqrt(float(np.mean(frame * frame)))
            db = 20.0 * math.log10(rms + 1e-10)
            voice_thr = max(self._running_max_db - self._voice_drop_db, self._floor_db + 6.0)
            if db >= voice_thr:
                self._voiced_frames += 1
            peak_time = self._feed_frame(self._frame_counter, db)
            self._frame_counter += 1
            if peak_time is not None:
                peaks.append(peak_time)
        self._carry = buf[n_frames * fl :].copy()
        return peaks

    def _feed_frame(self, gidx: int, db: float) -> float | None:
        """新フレームを取り込み、中央フレームが音節核なら時刻を返す。"""
        peak_time: float | None = None
        if self._f0 is not None and self._f1 is not None:
            peak_time = self._evaluate_center(self._f0, self._f1, (gidx, db))
        self._f0 = self._f1
        self._f1 = (gidx, db)
        return peak_time

    def _evaluate_center(
        self,
        left: tuple[int, float],
        center: tuple[int, float],
        right: tuple[int, float],
    ) -> float | None:
        """3フレームの中央 ``center`` を音節核候補として評価する。"""
        cidx, cdb = center
        # 走行最大強度をゆるやかに減衰させつつ更新（話声量の変化に追従）。
        self._running_max_db = max(cdb, self._running_max_db - self._max_decay_per_frame)
        threshold = max(self._running_max_db - self._drop_db, self._floor_db)

        is_local_max = cdb >= left[1] and cdb >= right[1]
        accepted: float | None = None
        if is_local_max and cdb >= threshold:
            gap_ok = (cidx - self._last_peak_idx) * self._hop_sec >= self._min_dist_sec
            dip_ok = (cdb - self._min_since_peak) >= self._dip_db
            if gap_ok and dip_ok:
                accepted = (cidx + 0.5) * self._hop_sec  # フレーム中心の時刻
                self._last_peak_idx = cidx
                self._min_since_peak = cdb

        # 次のピークに向けた谷（最小強度）を更新。
        self._min_since_peak = min(self._min_since_peak, cdb)
        return accepted
