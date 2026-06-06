"""漢字混じりテキストをかなに変換し、モーラ数を数える。

発話速度の正規単位を「モーラ」に統一するためのモジュール。
Vosk は漢字混じりのテキストを返すため（例: ``東京駅`` = 3文字だが6モーラ）、
そのままの文字数では信号処理側のモーラ数とも、アナウンサー基準（≒モーラ/分）とも
噛み合わない。ここで pykakasi を使ってひらがなに直し、日本語のモーラ規則で数える。

モーラ規則（おおまかに）:
- かな・カタカナ・長音「ー」は各1モーラ。
- 促音「っ」・撥音「ん」は各1モーラ。
- 小書きの拗音「ゃゅょ」や小書き母音「ぁぃぅぇぉ」は直前の字と合体し、単独では0。
- 記号・英数字・空白はモーラに数えない。
"""

from __future__ import annotations

import functools

import pykakasi

# 直前の字と合体して1モーラを構成する小書きかな（拗音・小書き母音）。
# 促音「っ」「ッ」は1モーラなので含めない。
_SMALL_COMBINING: frozenset[str] = frozenset(
    "ぁぃぅぇぉゃゅょゎゕゖ"  # ひらがな小書き
    "ァィゥェォャュョヮヵヶ"  # カタカナ小書き
)

# 長音記号と、モーラとして数えるかなの反復記号。
_LONG_VOWEL: str = "ー"
_ITERATION_MARKS: frozenset[str] = frozenset("ゝゞヽヾ")


@functools.lru_cache(maxsize=1)
def _converter() -> pykakasi.kakasi:
    """pykakasi インスタンスは生成コストがあるため使い回す。"""
    return pykakasi.kakasi()


def _is_kana(ch: str) -> bool:
    """ひらがな（U+3041–U+3096）またはカタカナ（U+30A1–U+30FA）か。"""
    code = ord(ch)
    return 0x3041 <= code <= 0x3096 or 0x30A1 <= code <= 0x30FA


def to_hiragana(text: str) -> str:
    """漢字混じりテキストをひらがな主体の読みに変換する。

    pykakasi は MeCab 非依存の辞書ベースで文脈依存の読みには弱いが、
    「モーラ数の概算」という用途には十分。変換結果の ``hira`` を連結して返す。
    """
    if not text:
        return ""
    return "".join(part["hira"] for part in _converter().convert(text))


def count_morae(text: str) -> int:
    """漢字混じり（またはかな）テキストのモーラ数を返す。

    >>> count_morae("東京駅")
    6
    >>> count_morae("きょう")
    2
    >>> count_morae("学校")
    4
    >>> count_morae("コーヒー")
    4
    """
    total = 0
    for ch in to_hiragana(text):
        if ch in _SMALL_COMBINING:
            continue
        if _is_kana(ch) or ch == _LONG_VOWEL or ch in _ITERATION_MARKS:
            total += 1
    return total
