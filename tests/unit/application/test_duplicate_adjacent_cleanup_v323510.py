"""[v3.23.510] Dọn CẶP dòng liền kề bị gán cùng một bản dịch.

Khi một CÂU bị tách qua hai dòng phụ đề, model dịch trọn câu rồi đặt y hệt bản dịch vào cả hai
dòng (dữ liệu thật: 连大家的工资都 / 发不了 cùng ra "không phát nổi."). Ràng buộc số phần tử ở
schema (v3.23.509) khiến model buộc điền đủ số dòng nên LẶP dòng bên cạnh thay vì bỏ trống —
duplicate_adjacent đo được tăng từ 2 lên 10.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)


def _line(index: int, text: str) -> TranslationLine:
    return TranslationLine(index=index, start_ms=0, end_ms=100, text=text)


def _request() -> MagicMock:
    request = MagicMock()
    request.stages = [
        TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
    ]
    request.context = TranslationContext(target_lang="Vietnamese")
    request.video_refs = []
    return request


def _use_case(retried: list[TranslationLine] | Exception) -> tuple:
    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    translator = MagicMock()
    if isinstance(retried, Exception):
        translator.translate_stage.side_effect = retried
    else:
        translator.translate_stage.return_value = retried
    use_case._translator = translator
    return use_case, translator


CURRENT = [_line(1859, "Nhà máy khó khăn,"), _line(1860, "không phát nổi."),
           _line(1861, "không phát nổi.")]
SOURCE = [_line(1859, "厂里困难"), _line(1860, "连大家的工资都"), _line(1861, "发不了")]


class TestRepair:
    def test_duplicate_pair_retranslated(self) -> None:
        use_case, _ = _use_case(
            [_line(1860, "ngay cả lương mọi người"), _line(1861, "cũng không phát nổi.")]
        )
        result = use_case._cleanup_duplicate_adjacent(CURRENT, SOURCE, _request(), None)
        texts = {line.index: line.text for line in result}
        assert texts[1860] != texts[1861]
        assert texts[1859] == "Nhà máy khó khăn,"  # dòng không liên quan giữ nguyên

    def test_identical_source_left_alone(self) -> None:
        """Nguồn giống nhau thì dịch giống nhau là ĐÚNG — không được đụng vào."""
        use_case, translator = _use_case([])
        current = [_line(1, "Vâng."), _line(2, "Vâng.")]
        source = [_line(1, "好的"), _line(2, "好的")]
        result = use_case._cleanup_duplicate_adjacent(current, source, _request(), None)
        assert result is current
        assert not translator.translate_stage.called

    def test_no_duplicates_skips_model_call(self) -> None:
        use_case, translator = _use_case([])
        current = [_line(1, "Xin chào"), _line(2, "Tạm biệt")]
        source = [_line(1, "你好"), _line(2, "再见")]
        assert use_case._cleanup_duplicate_adjacent(current, source, _request(), None) is current
        assert not translator.translate_stage.called

    def test_empty_translations_ignored(self) -> None:
        """Hai dòng cùng RỖNG không phải lỗi trùng — vấn đề khác, do khâu khác xử lý."""
        use_case, translator = _use_case([])
        current = [_line(1, ""), _line(2, "")]
        source = [_line(1, "你好"), _line(2, "再见")]
        assert use_case._cleanup_duplicate_adjacent(current, source, _request(), None) is current
        assert not translator.translate_stage.called


class TestSafety:
    def test_keeps_original_when_retry_still_duplicated(self) -> None:
        """Dịch lại vẫn trùng → giữ nguyên bản cũ, không làm xấu thêm."""
        use_case, _ = _use_case(
            [_line(1860, "không phát nổi."), _line(1861, "không phát nổi.")]
        )
        result = use_case._cleanup_duplicate_adjacent(CURRENT, SOURCE, _request(), None)
        assert result is CURRENT

    def test_keeps_original_when_retry_fails(self) -> None:
        use_case, _ = _use_case(RuntimeError("mạng lỗi"))
        result = use_case._cleanup_duplicate_adjacent(CURRENT, SOURCE, _request(), None)
        assert result is CURRENT

    def test_no_literal_stage_keeps_original(self) -> None:
        use_case, translator = _use_case([])
        request = _request()
        request.stages = [
            TranslationStageConfig(kind=TranslationStageKind.STYLE, model_name="m")
        ]
        assert use_case._cleanup_duplicate_adjacent(CURRENT, SOURCE, request, None) is CURRENT
        assert not translator.translate_stage.called

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_retry_text_ignored(self, blank: str) -> None:
        """Bản dịch lại rỗng thì không ghi đè (thà giữ bản cũ còn hơn mất dòng)."""
        use_case, _ = _use_case([_line(1860, blank), _line(1861, "cũng không phát nổi.")])
        result = use_case._cleanup_duplicate_adjacent(CURRENT, SOURCE, _request(), None)
        texts = {line.index: line.text for line in result}
        assert texts[1860] == "không phát nổi."
