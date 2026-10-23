"""[v3.23.580] Mở lại dự án giữ giãn giờ đọc; số đọc "phẩy"; cờ trùng dòng theo trợ từ.

Chuẩn đoán thật v3.23.578 → v3.23.579:

* Cùng phim "傻子落水…" (3902 dòng): file xuất sau khi MỞ LẠI dự án báo 758 dòng đọc
  không kịp, file xuất ngay sau khi dịch chỉ 397-426. Phiên dịch chỉ lưu CHỮ, còn thời
  gian lấy lại từ phụ đề nguồn — phần giãn giờ đọc làm lúc dịch xong bị mất, và trang
  lồng tiếng nhận đúng các dòng đó.
* Phim "he thong sung the": "三十四点五万元" dịch "ba mươi bốn phẩy năm vạn tệ" (đúng
  345.000) bị lưới số đọc thành "năm vạn" (50.000) → báo sai giá trị; phim mà có cách đọc
  mẫu cho 50.000 thì lưới còn "sửa" giữa câu thành chữ vô nghĩa.
* "住手" / "住手啊" cùng ra "Dừng tay!" — dịch ĐÚNG, khâu dọn không đụng (v3.23.577),
  nhưng file chuẩn đoán vẫn báo ``duplicate_adjacent``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.number_consistency_guard import (
    find_number_findings,
    find_number_fixes,
    vietnamese_number_value,
)
from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
    source_content_key,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

# (câu gốc, bắt đầu, kết thúc gốc, bản dịch cuối, kết thúc sau khi giãn) — dòng 3435-3441
# thật của phim "傻子落水…" (v3.23.579).
READING_ROWS = [
    ("一起呀", 6753.733, 6754.133, "đi cùng luôn đi.", 6754.495),
    ("我就不去了", 6754.733, 6755.533, "Em không đi đâu.", 6755.533),
    ("我对车不感兴趣", 6756.267, 6756.933, "Em không có hứng thú với xe cộ,", 6757.567),
    ("我对钱比较感兴趣", 6757.667, 6758.6, "em có hứng thú với tiền hơn.", 6759.0),
    ("我得去银行看看", 6759.133, 6760.133, "Em phải ra ngân hàng xem thử", 6760.133),
    ("一千万长什么样", 6760.267, 6761.533, "mười triệu NDT trông như thế nào đã.",
     6761.967),
    ("哎小财迷", 6762.067, 6763.267, "Đúng là đồ ham tiền.", 6763.267),
]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _source() -> list[SubtitleEvent]:
    return [
        SubtitleEvent(index=i, text=row[0], interval=TimeInterval(row[1], row[2]))
        for i, row in enumerate(READING_ROWS, start=1)
    ]


def _view_model():
    from subtitles_extractor.composition.bootstrap import bootstrap_for_gui
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    container = bootstrap_for_gui()
    container.settings_service.current.translation.extend_display_for_reading = True
    container.settings_service.current.translation.reading_cps_budget = 21.0
    return TranslatePageViewModel(container)


class TestRestoredTranslationKeepsReadingTime:
    VIDEO = "/fake/reading_restore_v323580.mp4"

    def test_reopened_project_is_extended_like_a_fresh_translation(self, app) -> None:
        writer = _view_model()
        writer.set_source_events(_source())
        writer._last_translate_video_path = self.VIDEO
        writer._translated_events = [
            SubtitleEvent(index=i, text=row[3], interval=TimeInterval(row[1], row[2]))
            for i, row in enumerate(READING_ROWS, start=1)
        ]
        writer._persist_current_result()

        reader = _view_model()
        reader.set_source_events(_source())
        reader.restore_stage_outputs_for_video(self.VIDEO)
        ends = [round(event.end_sec, 3) for event in reader.translated_events]
        assert ends == [row[4] for row in READING_ROWS]

    def test_retranslated_line_is_extended_again(self, app) -> None:
        vm = _view_model()
        vm.set_source_events(_source())
        vm._translated_events = [
            SubtitleEvent(index=i, text="Được.", interval=TimeInterval(row[1], row[2]))
            for i, row in enumerate(READING_ROWS, start=1)
        ]
        vm._worker = None
        vm._retranslate_indices = [2]
        vm._on_retranslate_done(SimpleNamespace(events=[
            SubtitleEvent(index=3, text=READING_ROWS[2][3],
                          interval=TimeInterval(READING_ROWS[2][1], READING_ROWS[2][2]))
        ]))
        assert round(vm.translated_events[2].end_sec, 3) == READING_ROWS[2][4]
        assert round(vm.translated_events[1].end_sec, 3) == READING_ROWS[1][2]


class TestSpokenDecimal:
    @pytest.mark.parametrize(
        ("phrase", "value"),
        [
            ("ba mươi bốn phẩy năm vạn", 345_000),
            ("ba mươi tư phẩy năm vạn", 345_000),
            ("hai phẩy hai lăm triệu", 2_250_000),
            ("một phẩy năm tỷ", 1_500_000_000),
        ],
    )
    def test_values(self, phrase: str, value: int) -> None:
        assert vietnamese_number_value(phrase) == value

    def test_unclear_reading_is_skipped(self) -> None:
        assert vietnamese_number_value("ba phẩy mười lăm triệu") is None

    def test_real_lines_are_not_mismatches(self) -> None:
        sources = {656: "三十四点五万元", 663: "消费三十四点五万元"}
        translations = {
            656: "là ba mươi bốn phẩy năm vạn tệ.",
            663: "tiêu dùng ba mươi tư phẩy năm vạn tệ,",
        }
        assert [f for f in find_number_findings(sources, translations)
                if f.kind == "mismatch"] == []

    def test_whole_number_is_replaced_never_its_tail(self) -> None:
        # Phim có cách đọc mẫu cho 345.000 ở dòng khác: trước đây lưới thay "năm vạn" giữa
        # câu thành "là ba mươi bốn phẩy ba trăm bốn mươi lăm nghìn tệ". Nay cả cụm được
        # nhận là 345.000 lối "vạn" và thống nhất theo cách đọc của phim.
        sources = {663: "消费三十四点五万元", 656: "三十四点五万元"}
        translations = {
            663: "tiêu dùng ba trăm bốn mươi lăm nghìn tệ,",
            656: "là ba mươi bốn phẩy năm vạn tệ.",
        }
        # [v3.23.590] Một dòng 'vạn', một dòng hiện đại: bằng phiếu thì số dưới một triệu
        # theo lối 'vạn' — dòng 663 đổi, vẫn thay TRỌN cụm số.
        [fix] = find_number_fixes(sources, translations)
        assert fix.index == 663
        assert fix.text_after == "tiêu dùng ba mươi bốn phẩy năm vạn tệ,"
        assert fix.reason == "lệch lối đọc"


def _events(texts: list[str]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(index=i, text=text, start_sec=i, end_sec=i + 0.9, confidence=1.0)
        for i, text in enumerate(texts, start=1594)
    ]


class TestDuplicateAdjacentUsesContent:
    def test_real_particle_only_pair_is_not_a_duplicate(self) -> None:
        flags = detect_quality_flags(
            _events(["住手", "住手啊"]), _events(["Dừng tay!", "Dừng tay!"])
        )
        assert flags["duplicate_adjacent"] == []

    def test_different_content_is_still_flagged(self) -> None:
        flags = detect_quality_flags(
            _events(["住手", "你们别忘了"]), _events(["Dừng tay!", "Dừng tay!"])
        )
        assert flags["duplicate_adjacent"] == [{"indices": [1594, 1595]}]

    def test_cleanup_and_diagnostics_share_the_rule(self) -> None:
        from subtitles_extractor.application.services.line_echo_guard import (
            repeated_line_groups,
        )

        sources = {1594: "住手", 1595: "住手啊"}
        assert source_content_key("住手") == source_content_key("住手啊")
        assert repeated_line_groups(sources, {1594: "Dừng tay!", 1595: "Dừng tay!"}) == []
