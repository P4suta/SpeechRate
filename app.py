"""SpeechRate — リアルタイム発話速度フィードバック（Streamlit UI）。

起動: ``uv run streamlit run app.py``（または ``uv run main.py``）。

面接・原稿読み上げの練習用。話している最中にモーラ/分の速度メーターが見え、
早口（赤）／適正（緑）／ゆっくり（青）を即座にフィードバックする。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from speechrate import metrics, models
from speechrate.asr import load_model
from speechrate.audio import AudioEngine, list_input_devices

st.set_page_config(page_title="SpeechRate", page_icon="🎤", layout="wide")


@st.cache_resource(show_spinner="Vosk日本語モデルを準備中…（初回のみ約48MBをDL）")
def get_engine() -> AudioEngine:
    """Voskモデルと AudioEngine を1度だけ生成し、rerun 間で共有する。"""
    model_path = models.ensure_model()
    return AudioEngine(load_model(model_path))


def main() -> None:
    st.title("🎤 SpeechRate — リアルタイム発話速度")
    st.caption("面接・原稿読み上げの練習に。落ち着いたペース（目安 約300モーラ/分）を目指そう。")

    try:
        engine = get_engine()
    except models.ModelDownloadError as exc:
        st.error(str(exc))
        st.stop()

    low, high, live_window, smooth, device, realtime = _sidebar()
    _controls(engine, device, realtime)

    refresh = 0.5 if engine.running else None
    st.fragment(run_every=refresh)(lambda: _dashboard(engine, low, high, live_window, smooth))()


def _sidebar() -> tuple[float, float, float, int, int | None, bool]:
    with st.sidebar:
        st.header("設定")
        low, high = st.slider(
            "適正レンジ（モーラ/分）",
            min_value=100,
            max_value=600,
            value=(int(metrics.DEFAULT_TARGET_LOW), int(metrics.DEFAULT_TARGET_HIGH)),
            step=10,
            help="この範囲内なら『適正』。アナウンサーの聞き取りやすい速さ ~300 が目安。",
        )
        live_window = st.slider("ライブ窓（秒）", 2, 10, int(metrics.DEFAULT_LIVE_WINDOW))
        smooth = st.slider(
            "グラフ平滑化（移動平均の点数）",
            min_value=1,
            max_value=15,
            value=1,
            help="1で平滑化なし。大きいほどグラフのギザギザが滑らかになる（針の数値には影響しない）。",
        )

        st.divider()
        st.subheader("入力デバイス")
        devices = list_input_devices()
        labels = ["既定（システム入力）"] + [f"{idx}: {name}" for idx, name in devices]
        choice = st.selectbox("マイク", labels, index=0)
        device = None if choice == labels[0] else int(choice.split(":", 1)[0])

        realtime = st.checkbox(
            "WAVを実時間で再生", value=False, help="OFFなら最速で解析。ONなら録音と同じ速さで再生。"
        )
    return float(low), float(high), float(live_window), int(smooth), device, realtime


def _controls(engine: AudioEngine, device: int | None, realtime: bool) -> None:
    col_start, col_stop, col_clear = st.columns(3)
    with col_start:
        if st.button(
            "● 録音開始", type="primary", disabled=engine.running, use_container_width=True
        ):
            try:
                engine.start_microphone(device)
            except Exception as exc:  # マイク不可など。rerunせずエラーを表示し続ける。
                st.error(f"マイクを開始できませんでした: {exc}")
            else:
                st.rerun()
    with col_stop:
        if st.button("■ 停止", disabled=not engine.running, use_container_width=True):
            engine.stop()
            st.rerun()
    with col_clear:
        # rerun をまたいで残る前回の履歴（チャート・トランスクリプト・要約）を消す。
        if st.button("🗑 クリア", disabled=engine.running, use_container_width=True):
            engine.state.reset()
            st.rerun()

    with st.expander("WAVファイルを解析（マイク不要）"):
        uploaded = st.file_uploader("WAV（16bit PCM）", type=["wav"], disabled=engine.running)
        if uploaded is not None and st.button("解析を実行", disabled=engine.running):
            tmp = Path(tempfile.gettempdir()) / f"speechrate_{uploaded.name}"
            tmp.write_bytes(uploaded.getvalue())
            engine.start_file(str(tmp), realtime=realtime)
            st.rerun()


def _dashboard(
    engine: AudioEngine, low: float, high: float, live_window: float, smooth: int
) -> None:
    snap = engine.state.snapshot()
    now = snap.stream_time
    factor = metrics.calibration_factor(snap.total_syllables, snap.total_asr_morae)
    lr = metrics.live_rate(snap.syllable_times, now, live_window) * factor
    ar = metrics.asr_rate(snap.utterances, now)
    cls = metrics.classify(lr, low, high)

    status = "● 録音中" if engine.running else "■ 停止中"
    st.markdown(
        f"<div style='font-size:3rem;font-weight:700;color:{cls.color}'>"
        f"{lr:.0f} <span style='font-size:1.2rem;color:#666'>モーラ/分</span> "
        f"<span style='font-size:1.5rem'>— {cls.label}</span></div>"
        f"<div style='color:#888'>{status}</div>",
        unsafe_allow_html=True,
    )

    _level_meter(engine, snap)

    c1, c2, c3 = st.columns(3)
    calib = f"（ASR補正 ×{factor:.2f}）" if factor != 1.0 else ""
    c1.metric("ライブ速度（信号処理）", f"{lr:.0f}", help=f"低遅延の瞬間値{calib}")
    c2.metric("ASR速度（直近10秒）", f"{ar:.0f}", help="認識テキストからの安定値")
    c3.metric("経過時間", f"{now:.1f} 秒")

    _chart(snap, low, high, smooth)
    _transcript(snap)

    if not engine.running and now > 0:
        _summary(snap, low, high)


def _level_meter(engine: AudioEngine, snap) -> None:
    """入力レベルバー。ミュート/無接続なら一目で『信号ゼロ』と分かる。"""
    if not engine.running:
        return
    # ピーク振幅(0..1)は通常0.1〜0.5程度なので、見やすいよう3倍にスケールして表示。
    st.progress(min(1.0, snap.level * 3.0), text=f"🎙 入力レベル {snap.level:.2f}")
    if snap.level < 0.005:
        st.warning(
            "入力信号がほぼゼロです。マイクのミュート・接続・OSのマイク権限を確認してください。",
            icon="⚠️",
        )


def _chart(snap, low: float, high: float, smooth: int) -> None:
    if not snap.rate_history:
        return
    df = pd.DataFrame(snap.rate_history, columns=["秒", "モーラ/分"]).set_index("秒")
    if smooth > 1:
        df["モーラ/分"] = df["モーラ/分"].rolling(window=smooth, min_periods=1).mean()
    df["適正下限"] = low
    df["適正上限"] = high
    st.line_chart(df, height=220, color=["#1a73e8", "#bbbbbb", "#bbbbbb"])


def _transcript(snap) -> None:
    st.markdown("**認識テキスト**")
    text = "".join(u.text for u in snap.utterances)
    if snap.partial:
        text += f" _{snap.partial}_"
    with st.container(height=160, border=True):
        st.markdown(text or "_（認識待ち…マイクに向かって話してみよう）_")


def _summary(snap, low: float, high: float) -> None:
    s = metrics.summarize(snap, low, high)
    st.divider()
    st.subheader("📊 セッション要約")
    a, b, c = st.columns(3)
    a.metric("発話速度（ポーズ込み）", f"{s.speech_rate:.0f}", help="総モーラ ÷ 総時間")
    b.metric("調音速度（ポーズ除外）", f"{s.articulation_rate:.0f}", help="総モーラ ÷ 発声時間")
    c.metric("適正レンジ滞在率", f"{s.pct_in_range * 100:.0f}%")
    d, e, f = st.columns(3)
    d.metric("総時間 / 発声時間", f"{s.duration:.1f}s / {s.voiced_seconds:.1f}s")
    e.metric("ポーズ合計", f"{s.pause_seconds:.1f}s")
    f.metric("瞬間 最小〜最大", f"{s.min_rate:.0f}〜{s.peak_rate:.0f}")
    st.caption(
        f"総モーラ: 信号処理 {s.total_syllables} / ASR {s.total_asr_morae}　"
        f"（ASR速度 {s.asr_speech_rate:.0f} モーラ/分）"
    )


if __name__ == "__main__":
    main()
