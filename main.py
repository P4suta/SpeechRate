"""SpeechRate ランチャー。

``uv run main.py`` で Streamlit アプリ（app.py）を起動する。
直接 ``uv run streamlit run app.py`` と同等。
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli


def main() -> None:
    app = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
