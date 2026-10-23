"""[v3.23.553] Hồi quy cho độ chính xác lưới DỊCH THIẾU (chuẩn đoán thật v3.23.552).

Log thật (phim 3902 dòng): ``under_translation`` nhảy 6 → 30 và giai đoạn Bản địa hoá
hoàn nguyên 26 dòng, dù 26/26 dòng đó GIỐNG HỆT bản Tinh chỉnh. Ba nguyên nhân:

1. Trung vị tính RIÊNG cho bản trước và bản sau → dòng không đổi vẫn bị coi là "vừa hỏng".
2. Dấu câu bị tính vào độ dài gốc CJK ('怎么样?' = 4 thay vì 3 chữ Hán).
3. Tỉ lệ dòng ngắn bị lượng tử hoá, nằm đúng mép ngưỡng → lật cờ giữa các lần chạy.
"""

from __future__ import annotations

from subtitles_extractor.application.services.under_translation_guard import (
    find_regressions,
    find_under_translated,
)

_BASE_SOURCE = "你今天来这里做什么"
_BASE_TRANSLATION = "Hôm nay cháu tới chỗ này để làm gì vậy"  # 29 chữ cái / 9 chữ Hán


def _base_pairs(count: int = 20) -> list[tuple[int, str, str]]:
    return [(i, _BASE_SOURCE, _BASE_TRANSLATION) for i in range(1, count + 1)]


class TestSharedMedian:
    def test_unchanged_lines_are_never_regressions(self) -> None:
        """Ca thật: bản sau viết DÀI HƠN ở đa số dòng làm trung vị bản sau tăng.

        Dòng ngắn giữ nguyên chữ nào cũng không được bị hoàn nguyên.
        """
        longer = _BASE_TRANSLATION + " thế hả cháu ơi"
        items = [(i, _BASE_SOURCE, _BASE_TRANSLATION, longer) for i in range(1, 21)]
        items.append((1810, "你到底怎么样了", "Rốt cuộc cháu sao rồi?",
                      "Rốt cuộc cháu sao rồi?"))
        assert find_regressions(items) == set()

    def test_line_that_got_longer_is_not_regression(self) -> None:
        items = [(i, _BASE_SOURCE, _BASE_TRANSLATION, _BASE_TRANSLATION)
                 for i in range(1, 21)]
        items.append((50, "不是叔叔不给你机会", "Cho cơ hội,", "Cho cháu cơ hội,"))
        assert find_regressions(items) == set()

    def test_real_shortening_is_still_caught(self) -> None:
        items = [(i, _BASE_SOURCE, _BASE_TRANSLATION, _BASE_TRANSLATION)
                 for i in range(1, 21)]
        items.append((84, "所以我才跟着大家", "Nên cháu mới theo mọi người",
                      "Nên cháu hùa,"))
        assert find_regressions(items) == {84}


class TestPunctuationIgnored:
    def test_punctuation_does_not_change_verdict(self) -> None:
        plain = find_under_translated(
            [*_base_pairs(), (99, "你到底想干什么", "Rốt cuộc muốn gì")]
        )
        punctuated = find_under_translated(
            [*_base_pairs(), (99, "你到底想干什么\uff1f\uff01", "Rốt cuộc muốn gì?!")]
        )
        assert [s.index for s in plain] == [s.index for s in punctuated]

    def test_short_source_with_punctuation_is_excluded(self) -> None:
        """'怎么样?' chỉ có 3 chữ Hán — quá ngắn để thống kê, dù có dấu câu."""
        suspects = find_under_translated([*_base_pairs(), (1810, "怎么样?", "Thế nào?")])
        assert suspects == []


class TestShortfallMargin:
    def test_knife_edge_line_not_flagged(self) -> None:
        """Thiếu vài chữ cái so với kỳ vọng là nhiễu lượng tử hoá, chưa phải mất nghĩa."""
        # Kỳ vọng = 0.5 * (29/9) * 4 ≈ 6.4 chữ cái; bản dịch 5 chữ cái — trong biên.
        suspects = find_under_translated([*_base_pairs(), (7, "你说什么", "Nói gì?")])
        assert suspects == []

    def test_real_negation_loss_still_flagged(self) -> None:
        suspects = find_under_translated(
            [*_base_pairs(), (91, "不是叔叔不给你机会", "Cho cơ hội,")]
        )
        assert [s.index for s in suspects] == [91]

    def test_latin_job_measure_unchanged(self) -> None:
        """Nguồn Latin giữ cách đo cũ (ngưỡng 0,35 hiệu chỉnh theo đó), không biên."""
        base = [(i, "What are you doing here today?", "Hôm nay cháu tới đây làm gì?")
                for i in range(1, 20)]
        suspects = find_under_translated(
            [*base, (5, "I never said that to you, okay?", "Ừ.")]
        )
        assert [s.index for s in suspects] == [5]


class TestRepetitiveSource:
    def test_laughter_is_excluded(self) -> None:
        """Ca thật (4/5 lần chạy): '哈哈哈哈哈哈哈' → 'Hahahaha...' bị cờ oan."""
        suspects = find_under_translated(
            [*_base_pairs(), (2679, "哈哈哈哈哈哈哈", "Hahahaha...")]
        )
        assert suspects == []

    def test_real_fragment_loss_still_flagged(self) -> None:
        """Mất '神仙姐姐' KHÔNG được dòng kề bù — phải còn bị bắt với biên 2 chữ cái."""
        suspects = find_under_translated(
            [*_base_pairs(), (217, "这就是神仙姐姐的", "đây chính là...")]
        )
        assert [s.index for s in suspects] == [217]


class TestNeighbourAbsorption:
    _REORDERED: tuple[tuple[int, str, str], ...] = (
        (216, "难道", "Chẳng lẽ"),
        (217, "这就是神仙姐姐的", "đây chính là..."),
        (218, "元神碎片", "mảnh nguyên thần của Thần Tiên Tỷ Tỷ?"),
    )

    def test_content_moved_to_next_line_not_flagged(self) -> None:
        """Ca thật v3.23.552: nội dung dời sang dòng 218 — không mất gì."""
        assert find_under_translated([*_base_pairs(), *self._REORDERED]) == []

    def test_regression_guard_does_not_split_reordered_pair(self) -> None:
        before = {
            216: "Chẳng lẽ",
            217: "đây chính là mảnh nguyên thần",
            218: "của Thần Tiên Tỷ Tỷ?",
        }
        items = [(i, _BASE_SOURCE, _BASE_TRANSLATION, _BASE_TRANSLATION)
                 for i in range(1, 21)]
        items += [(i, src, before[i], after) for i, src, after in self._REORDERED]
        assert find_regressions(items) == set()

    def test_normal_neighbours_do_not_hide_omission(self) -> None:
        pairs = [*_base_pairs(), (21, "不是叔叔不给你机会", "Cho cơ hội,")]
        assert [s.index for s in find_under_translated(pairs)] == [21]
