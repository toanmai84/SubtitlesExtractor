"""[v3.23.707] Dựng phụ đề: chữ bị rơi ở MỘT PHẦN khung phải được khôi phục cả khi bộ gom đã tách thành nhiều group.

Lượt chạy thật của chủ dự án (phim nối 86 tập, 208 752 khung, bước 0,03 s, có tiền xử lý ảnh): 10 câu sai chữ so với
đáp án đã kiểm bằng ảnh (3663 câu). 5 câu trong đó VẪN có một phần khung đọc đúng, nhưng các khung xen kẽ bị tách thành
nhiều group thuần nhất rồi gộp theo số đông, nên bộ khôi phục chữ không bao giờ thấy bằng chứng.
Chuỗi khung dưới đây chép từ dữ liệu thô đó.
"""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.subtitle_pipeline.event_filters import (
    _restore_across_merged,
    _restore_dash_as_yi,
)
from tests.unit.application import test_ocr_line_boundaries_v323672 as lines
from tests.unit.application.test_ocr_line_boundaries_v323672 import _build

STEP = 0.03


def _runs(start: float, runs: list[tuple[str, int]], conf: float = 0.99) -> list[tuple[float, float, str, float]]:
    """Chuỗi ``(chữ, số khung)`` liền nhau → timeline ``(start, end, text, conf)`` cho ``_build``."""
    timeline, cursor = [], start
    for text, frames in runs:
        end = cursor + (frames - 1) * STEP
        timeline.append((round(cursor, 2), round(end, 2), text, conf))
        cursor = end + STEP
    return timeline


class TestRealSequences:
    @pytest.mark.parametrize(("runs", "expected"), [
        ([("天都拦不住", 12), ("一天都拦不住", 7), ("天都拦不住", 22), ("一天都拦不住", 4)], "一天都拦不住"),
        ([("百雷劫太重了", 28), ("一百雷劫太重了", 7)], "一百雷劫太重了"),
        ([("三百道雷劫", 8), ("百道雷劫", 16), ("三百道雷劫", 4)], "三百道雷劫"),
    ])
    def test_dropped_char_is_restored_across_groups(self, runs: list[tuple[str, int]], expected: str) -> None:
        assert _build(_runs(10.0, runs), step=STEP) == [expected]

    def test_dash_read_of_leading_yi_is_restored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """"一天呐": "-天呐" / "天呐" xen kẽ. [v3.23.718] Hình học như ảnh thật: câu ba chữ đúng tâm; bản đọc thiếu "一"
        chỉ phủ hai ô bên phải nên lệch phải nửa chữ (+20 px ở bộ 3.23.690, +26 px ở bộ 3.23.711)."""
        right_edge = 301 + lines.CHAR_W * 3 // 2

        def box(text: str, conf: float) -> lines.OcrTextBox:
            if text in ("-天呐", "天呐"):
                # Câu "一天呐" (3 ô, mép phải cố định): bản đọc chỉ phủ hai ô bên phải; dấu gạch chỉ là mẩu nhỏ
                # (đo thật bộ 3.23.690: hộp "-天呐" tâm 314, "天呐" tâm 314–316, tâm phụ đề 294).
                x1 = right_edge - lines.CHAR_W * 2 - (12 if text.startswith("-") else 0)
                width = right_edge - x1
            else:                                              # câu ngữ cảnh: đúng tâm
                x1 = 301 - lines.CHAR_W * len(text) // 2
                width = lines.CHAR_W * len(text)
            return lines.OcrTextBox(text=text, confidence=lines.Confidence(conf),
                                    polygon=[(x1, 18), (x1 + width, 18), (x1 + width, 90), (x1, 90)])

        monkeypatch.setattr(lines, "_box", box)
        runs = [("-天呐", 4), ("天呐", 8), ("-天呐", 3), ("天呐", 9)]
        assert _build(_runs(10.0, runs), step=STEP) == ["一天呐"]

    def test_dash_added_to_a_centred_line_is_not_yi(self) -> None:
        """[v3.23.718] OCR thêm "-" vào một phần lần đọc của câu ĐÚNG TÂM ("天呐" thật): không thêm "一"."""
        runs = [("-天呐", 4), ("天呐", 8), ("-天呐", 3), ("天呐", 9)]
        assert _build(_runs(10.0, runs), step=STEP) == ["天呐"]

    def test_no_evidence_no_change(self) -> None:
        # "个万年炼气的人": không khung nào đọc ra "一" → không bịa.
        assert _build(_runs(10.0, [("个万年炼气的人", 35)]), step=STEP) == ["个万年炼气的人"]

    def test_stable_distinct_lines_stay_separate(self) -> None:
        # Hai câu thật mỗi câu đứng yên lâu ("你不要抛弃我" → "不要抛弃我", mỗi câu ≥ 0,7 s) không bị gộp.
        runs = [("你不要抛弃我", 30), ("不要抛弃我", 30)]
        assert _build(_runs(10.0, runs), step=STEP) == ["你不要抛弃我", "不要抛弃我"]


class TestRestoreAcrossMerged:
    def test_uses_all_merged_groups(self) -> None:
        evidence = {"天都拦不住": [(12, 0.99), (22, 0.99)], "一天都拦不住": [(7, 0.99), (4, 0.99)]}
        assert _restore_across_merged("天都拦不住", evidence) == "一天都拦不住"

    def test_low_confidence_or_single_text_is_ignored(self) -> None:
        assert _restore_across_merged("天都拦不住", {"天都拦不住": [(30, 0.99)]}) == "天都拦不住"
        evidence = {"天都拦不住": [(30, 0.99)], "一天都拦不住": [(3, 0.5)]}
        assert _restore_across_merged("天都拦不住", evidence) == "天都拦不住"

    def test_non_yi_char_needs_twenty_percent(self) -> None:
        # Chữ khác "一" phải có ≥ 20 % số khung (luật sẵn có từ v3.23.652): 2/42 khung → không chèn.
        evidence = {"终于被我逮到了吧": [(40, 0.99)], "终于被我逮到了吧天": [(2, 0.99)]}
        assert _restore_across_merged("终于被我逮到了吧", evidence) == "终于被我逮到了吧"


class TestDashAsYi:
    def test_needs_both_forms(self) -> None:
        mixed = [(i * STEP, "-天呐" if i < 8 else "天呐") for i in range(25)]
        assert _restore_dash_as_yi("天呐", mixed) == "一天呐"
        # Gạch đầu dòng hội thoại: MỌI khung đều có gạch → không đụng.
        assert _restore_dash_as_yi("你好", [(i * STEP, "-你好") for i in range(25)]) == "你好"

    def test_too_few_dashed_frames(self) -> None:
        few = [(i * STEP, "-天呐" if i < 2 else "天呐") for i in range(25)]
        assert _restore_dash_as_yi("天呐", few) == "天呐"

    def test_other_shapes_are_untouched(self) -> None:
        mixed = [(i * STEP, "-天呐" if i < 8 else "天呐") for i in range(25)]
        assert _restore_dash_as_yi("一天呐", mixed) == "一天呐"
        assert _restore_dash_as_yi("天呐\n不过师叔", mixed) == "天呐\n不过师叔"
        assert _restore_dash_as_yi("", mixed) == ""
        assert _restore_dash_as_yi("OK", [(0.0, "-OK"), (0.03, "OK"), (0.06, "-OK"), (0.09, "-OK")]) == "OK"
