"""Tests bảo vệ v3.7 SubtitleBuilder quality fix (đối chiếu 12 bộ test data).

FIX-PERCENT:  ``clean_edge_noise`` KHÔNG được strip ``%`` ở biên cuối. Trước đây
              ``30%`` → ``30`` (mất nghĩa phần trăm). Đo trên 12 bộ: +1 câu khớp
              (test2), KHÔNG bộ nào giảm ⇒ an toàn giữ.

GHI CHÚ — vì sao KHÔNG có fix prefix '一':
    Đã thử giới hạn ``prefix_prepend`` chỉ cho '一' (trên test4 thấy +5 do chặn
    artifact '二'/'心'/'不'). Nhưng đo lại trên TOÀN BỘ 12 bộ test cho thấy thay
    đổi đó NET ÂM (-4) — nó phá các phục hồi non-'一' ĐÚNG ở test3/test1 nhiều
    hơn số artifact chặn được. Theo nguyên tắc "chỉ sửa khi tăng chất lượng tổng
    thể", đã GIỮ NGUYÊN hành vi gốc. Test dưới xác nhận hành vi gốc vẫn còn.
"""

from __future__ import annotations

from subtitles_extractor.application.services.subtitle_pipeline.text_correction import (
    apply_yi_suffix_restore,
    clean_edge_noise,
)


class TestPercentNotStripped:
    """Fix duy nhất được giữ: KHÔNG strip '%' ở biên cuối."""

    def test_trailing_percent_preserved(self) -> None:
        assert clean_edge_noise("灵根纯度提升30%") == "灵根纯度提升30%"
        assert clean_edge_noise("亲和度提升50%") == "亲和度提升50%"

    def test_percent_with_space_preserved(self) -> None:
        assert clean_edge_noise("灵根纯度提升 30%") == "灵根纯度提升 30%"

    def test_genuine_trailing_junk_still_stripped(self) -> None:
        assert clean_edge_noise("正常文本~~~") == "正常文本"
        assert clean_edge_noise("文本@@") == "文本"

    def test_leading_junk_still_stripped(self) -> None:
        assert clean_edge_noise("...开始") == "开始"


class TestPrefixPrependBehaviorPreserved:
    """Xác nhận prefix_prepend vẫn hoạt động cho cả '一' lẫn non-'一'.

    [v3.7.3] Non-'一' prefix nay cần evidence mạnh hơn (>= 2 high-conf) để chống
    artifact rìa trái (logo/watermark OCR chèn nhầm ký tự đầu câu — đo trên cặp
    ground-truth test4 cho thấy net-dương). '一' đầu câu vẫn chỉ cần 1 high-conf
    vì drop thật rất phổ biến. Test này là regression guard: KHÔNG được tắt hẳn
    non-'一' prefix, chỉ siết ngưỡng.
    """

    def test_yi_prefix_still_restored(self) -> None:
        voted = "起去前厅用饭"
        candidates = [(voted, 1.0)] * 10 + [("一起去前厅用饭", 0.95)] * 1
        assert apply_yi_suffix_restore(voted, candidates) == "一起去前厅用饭"

    def test_non_yi_prefix_still_restored(self) -> None:
        voted = "愧是美人录里的人"
        # [v3.7.3] Non-'一' prefix vẫn phục hồi nhưng cần evidence MẠNH hơn
        # (>= 2 high-conf) để chống artifact rìa trái. 2 frame conf 0.95 = đủ.
        candidates = [(voted, 1.0)] * 8 + [("不愧是美人录里的人", 0.95)] * 2
        assert apply_yi_suffix_restore(voted, candidates) == "不愧是美人录里的人"
