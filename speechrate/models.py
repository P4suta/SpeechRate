"""Vosk 日本語モデルの自動ダウンロードとキャッシュ。

初回のみ ``vosk-model-small-ja-0.22``（約48MB, Apache 2.0）を取得して
プロジェクト直下の ``models/`` に展開する。以降はキャッシュを再利用する。
ネットワーク/プロキシで失敗した場合は、手動配置の手順を含む例外を送出する。
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

MODEL_NAME = "vosk-model-small-ja-0.22"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"

# プロジェクト直下の models/（このファイルは speechrate/ にあるので2つ上がリポジトリ root）。
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# 進捗コールバック: (ダウンロード済みバイト, 総バイト or None) を受け取る。
ProgressCallback = Callable[[int, int | None], None]


class ModelDownloadError(RuntimeError):
    """モデルの取得・展開に失敗したことを表す。手動配置の手順を含む。"""


def _model_dir(models_dir: Path) -> Path:
    return models_dir / MODEL_NAME


def is_model_present(models_dir: Path | None = None) -> bool:
    """モデルが展開済みでそのまま使える状態か。"""
    target = _model_dir(models_dir or DEFAULT_MODELS_DIR)
    # Vosk モデルは am/ conf/ などのサブディレクトリを含む。空でなければ使えるとみなす。
    return target.is_dir() and any(target.iterdir())


def ensure_model(
    models_dir: Path | None = None,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """モデルの展開先パスを返す。無ければダウンロードして展開する。

    :raises ModelDownloadError: ダウンロードまたは展開に失敗したとき。
    """
    base = models_dir or DEFAULT_MODELS_DIR
    target = _model_dir(base)
    if is_model_present(base):
        return target

    base.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / f"{MODEL_NAME}.zip"
            _download(MODEL_URL, zip_path, progress)
            _extract(zip_path, base)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        raise ModelDownloadError(
            "Vosk日本語モデルの自動取得に失敗しました。\n"
            f"  原因: {exc}\n"
            "手動でセットアップする場合:\n"
            f"  1) {MODEL_URL} をダウンロード\n"
            f"  2) 展開して '{target}' に配置（中に am/ や conf/ が見える状態）\n"
        ) from exc

    if not is_model_present(base):
        raise ModelDownloadError(
            f"展開は完了しましたが '{target}' が空です。アーカイブ構成を確認してください。"
        )
    return target


def _download(url: str, dest: Path, progress: ProgressCallback | None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "speechrate/0.1"})
    with urllib.request.urlopen(request) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        downloaded = 0
        if progress:
            progress(0, total)
        with dest.open("wb") as out:
            while chunk := response.read(1 << 16):
                out.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)


def _extract(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)
    # アーカイブのトップが MODEL_NAME 以外でも、それらしいディレクトリを正規名へ寄せる。
    expected = _model_dir(dest_dir)
    if expected.is_dir():
        return
    candidates = [p for p in dest_dir.iterdir() if p.is_dir() and p.name.startswith("vosk-model")]
    if len(candidates) == 1:
        shutil.move(str(candidates[0]), str(expected))
