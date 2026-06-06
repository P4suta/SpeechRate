"""マイク／WAV 入力エンジン（AudioEngine）。

リアルタイムの土台。設計（Streamlit のスレッドガイドラインに準拠）:

1. sounddevice のコールバック（PortAudioスレッド）は int16 バイトを ``queue.Queue`` に
   入れるだけ（高速・非ブロッキング）。
2. ワーカースレッドがキューから取り出し、Vosk（asr）と音節核検出（syllable）に流し、
   :class:`~speechrate.state.SharedState` を更新する。
3. UIスレッドは ``state.snapshot()`` を読むだけ。背景スレッドは Streamlit API を呼ばない。

WAV ファイルモード（:meth:`start_file`）も同じ処理経路を共有し、検出器の検証・
マイク無しデモ・テストを兼ねる。サンプルレートは Vosk 用に 16kHz へ揃える
（デバイスが 16kHz を拒否したらデバイス既定で開いて線形補間でリサンプル）。
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import vosk

from speechrate.asr import Recognizer
from speechrate.metrics import calibration_factor, live_rate
from speechrate.mora import count_morae
from speechrate.state import AsrUtterance, SharedState
from speechrate.syllable import SyllableDetector

PROCESS_SR = 16000  # Vosk 日本語モデルが想定するサンプルレート
BLOCK_SEC = 0.25  # 1ブロックの長さ（秒）


def resample_int16(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """int16 モノラルを線形補間でリサンプルする（簡易・低コスト）。"""
    if src_sr == dst_sr or audio.size == 0:
        return audio
    n_dst = max(1, round(audio.size * dst_sr / src_sr))
    x_old = np.arange(audio.size)
    x_new = np.linspace(0, audio.size - 1, n_dst)
    return np.interp(x_new, x_old, audio.astype(np.float32)).astype(np.int16)


def list_input_devices() -> list[tuple[int, str]]:
    """利用可能な入力デバイスの (index, name) 一覧。"""
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append((idx, info.get("name", f"device {idx}")))
    return devices


class AudioEngine:
    """マイク／WAV から発話速度イベントを生成する常駐エンジン。"""

    def __init__(self, model: vosk.Model) -> None:
        self._model = model
        self._state = SharedState()
        self._detector = SyllableDetector(PROCESS_SR)
        self._recognizer: Recognizer | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stream: sd.RawInputStream | None = None
        self._worker: threading.Thread | None = None
        self._file_thread: threading.Thread | None = None
        self._device_sr = PROCESS_SR
        self._running = False
        self._finalized = True
        self._mode = "idle"
        self._lock = threading.Lock()

    # ---- 公開プロパティ -------------------------------------------------
    @property
    def state(self) -> SharedState:
        return self._state

    @property
    def running(self) -> bool:
        return self._running

    @property
    def mode(self) -> str:
        return self._mode

    # ---- セッション管理 -------------------------------------------------
    def _begin_session(self, mode: str) -> None:
        self._detector.reset()
        self._state.reset()
        self._recognizer = Recognizer(self._model, PROCESS_SR)
        self._queue = queue.Queue()
        self._running = True
        self._finalized = False
        self._mode = mode

    def _finalize_session(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        # 認識器の後始末を済ませてから running を倒す。
        # （観測側が running=False を見た時点でネイティブ処理が終わっていることを保証）
        if self._recognizer is not None:
            final = self._recognizer.flush()
            if final:
                self._state.add_utterance(
                    AsrUtterance(self._detector.stream_time, final, count_morae(final))
                )
        self._state.set_partial("")
        self._mode = "idle"
        self._running = False

    # ---- マイク入力 -----------------------------------------------------
    def start_microphone(self, device: int | None = None) -> None:
        with self._lock:
            if self._running:
                return
            self._begin_session("mic")
            try:
                self._device_sr = self._open_stream(device)
            except Exception:
                self._running = False
                self._finalized = True
                self._mode = "idle"
                raise
            self._worker = threading.Thread(target=self._run_mic_worker, daemon=True)
            self._worker.start()

    def _open_stream(self, device: int | None) -> int:
        last_error: Exception | None = None
        for target in (PROCESS_SR, None):
            try:
                sr = target or int(sd.query_devices(device, "input")["default_samplerate"])
                stream = sd.RawInputStream(
                    samplerate=sr,
                    blocksize=int(sr * BLOCK_SEC),
                    dtype="int16",
                    channels=1,
                    device=device,
                    callback=self._callback,
                )
                stream.start()
                self._stream = stream
                return sr
            except Exception as exc:  # 次のサンプルレート候補へフォールバック
                last_error = exc
        raise RuntimeError(f"マイク入力ストリームを開けませんでした: {last_error}")

    def _callback(self, indata, frames, time_info, status) -> None:
        # PortAudioスレッド: 重い処理は禁止。バイトをキューへ入れるだけ。
        # frames / time_info / status は sounddevice のコールバック規約により受け取る。
        _ = (frames, time_info, status)
        self._queue.put(bytes(indata))

    def _run_mic_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            self._process_block(item, self._device_sr)

    # ---- WAV ファイル入力 ----------------------------------------------
    def start_file(self, path: str, *, realtime: bool = False) -> None:
        with self._lock:
            if self._running:
                return
            self._begin_session("file")
            self._file_thread = threading.Thread(
                target=self._run_file, args=(path, realtime), daemon=True
            )
            self._file_thread.start()

    def _run_file(self, path: str, realtime: bool) -> None:
        try:
            with sf.SoundFile(path) as f:
                file_sr = f.samplerate
                block = max(1, int(file_sr * BLOCK_SEC))
                for chunk in f.blocks(blocksize=block, dtype="int16", always_2d=True):
                    if not self._running:
                        break
                    mono = (
                        chunk[:, 0] if chunk.shape[1] == 1 else chunk.mean(axis=1).astype(np.int16)
                    )
                    self._process_block(mono.tobytes(), file_sr)
                    if realtime:
                        time.sleep(block / file_sr)
        finally:
            self._finalize_session()

    # ---- 共有処理経路 ---------------------------------------------------
    def _process_block(self, pcm16: bytes, src_sr: int) -> None:
        audio = np.frombuffer(pcm16, dtype=np.int16)
        if src_sr != PROCESS_SR:
            audio = resample_int16(audio, src_sr, PROCESS_SR)
            pcm16 = audio.tobytes()
        floats = audio.astype(np.float32) / 32768.0

        # 入力レベルメーター用（ピーク振幅 0..1）。ミュート/無接続なら 0 になる。
        self._state.set_level(float(np.max(np.abs(floats))) if floats.size else 0.0)

        # 信号処理（ライブ針）
        peaks = self._detector.process(floats)
        self._state.add_syllables(peaks)

        # 音声認識（正確な値＋テキスト）
        if self._recognizer is not None:
            final_text = self._recognizer.accept(pcm16)
            if final_text:
                self._state.add_utterance(
                    AsrUtterance(self._detector.stream_time, final_text, count_morae(final_text))
                )
            self._state.set_partial(self._recognizer.partial())

        # クロック更新＋速度履歴（チャート用、ASRで補正済みの値を記録）
        st = self._detector.stream_time
        self._state.update_clock(st, self._detector.voiced_seconds)
        snap = self._state.snapshot()
        factor = calibration_factor(snap.total_syllables, snap.total_asr_morae)
        self._state.push_rate(st, live_rate(snap.syllable_times, st) * factor)

    # ---- 停止 -----------------------------------------------------------
    def stop(self) -> None:
        with self._lock:
            if not self._running and self._finalized:
                return
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                finally:
                    self._stream = None
                self._queue.put(None)
                if self._worker is not None:
                    self._worker.join(timeout=2.0)
                    self._worker = None
            if self._file_thread is not None:
                self._file_thread.join(timeout=2.0)
                self._file_thread = None
            self._finalize_session()
