"""モーラ計数の検証。漢字混じり→かな→モーラ数。"""

from __future__ import annotations

import pytest

from speechrate.mora import count_morae

CASES = [
    ("東京駅", 6),  # とうきょうえき
    ("きょう", 2),  # 拗音は前の字と合体
    ("学校", 4),  # がっこう（促音は1モーラ）
    ("コーヒー", 4),  # 長音は1モーラ
    ("面接", 4),  # めんせつ（撥音は1モーラ）
    ("こんにちは", 5),
    ("切手", 3),  # きって
    ("ありがとうございます", 10),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_count_morae(text: str, expected: int) -> None:
    assert count_morae(text) == expected


def test_empty_is_zero() -> None:
    assert count_morae("") == 0
    assert count_morae("   ") == 0


def test_punctuation_and_ascii_ignored() -> None:
    assert count_morae("あ、い。 ABC 123") == 2
