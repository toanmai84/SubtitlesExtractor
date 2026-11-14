"""[v3.23.698] Mô hình thời lượng đọc tự nhiên khớp lại trên lượt lồng tiếng thật cả phim 86 tập (VieNeu, 14 giọng, 3664 dòng).

Ba dòng thật dưới đây VieNeu phải nén 1,90–1,95× (đo trong CSV debug của chủ dự án, 10/11/2026); mô hình cũ (141 dòng một
giọng) dự đoán các dòng này vừa khung nên không gợi ý rút gọn trước khi tổng hợp.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.infrastructure.tts.tts_fit import (
    NATURAL_BASE_S,
    NATURAL_PER_SYLLABLE_S,
    predict_misfits,
)

#: (chữ, bắt đầu, kết thúc, bắt đầu câu sau, tốc độ VieNeu thật phải dùng)
REAL_LINES = [
    ("lão tử vừa bước ra khỏi sơn môn", 55.240, 56.360, 56.560, 1.898),
    ("nhưng cái giá để trở nên mạnh mẽ", 73.400, 74.479, 74.480, 1.917),
    ("hai chúng ta cứ mỗi người gọi một kiểu,", 241.800, 242.880, 242.920, 1.950),
]


def test_model_matches_measured_fit() -> None:
    assert (NATURAL_BASE_S, NATURAL_PER_SYLLABLE_S) == (0.235, 0.247)


@pytest.mark.parametrize(("text", "start", "end", "next_start", "speed"), REAL_LINES)
def test_real_too_fast_lines_are_predicted(text, start, end, next_start, speed) -> None:
    events = [
        SimpleNamespace(start_sec=start, end_sec=end, text=text),
        SimpleNamespace(start_sec=next_start, end_sec=next_start + 1.0, text="ừ"),
    ]
    flagged = predict_misfits(events)
    assert [f.position for f in flagged] == [0]
    assert flagged[0].target_syllables < flagged[0].syllables
