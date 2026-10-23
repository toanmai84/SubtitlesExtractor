# ruff: noqa: RUF001 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.632] Vá RIÊNG dòng còn bị nghi DỊCH THIẾU ở bản cuối.

Ca thật (nhật ký phim 86 tập): ``师父师姐`` → ``Sư tỷ...`` (mất "sư phụ") và ``就把这个臭小子``
bị lưới báo ở CẢ BA giai đoạn rồi nằm nguyên trong bản cuối, vì lưới chỉ cảnh báo.
"""

from __future__ import annotations

from typing import Any

import pytest

from subtitles_extractor.application.services.translation_process_log import get_process_log
from subtitles_extractor.application.services.under_translation_guard import (
    find_under_translated,
)
from subtitles_extractor.application.services.under_translation_patch import (
    MAX_PATCH_LINES,
    accept_patches,
    patch_targets,
    patch_window,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)

_HAN = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈剑号巨阙珠称夜光"


def _normal_line(i: int) -> tuple[int, str, str]:
    """Dòng bình thường: 8 chữ Hán khác nhau → bản dịch ~28 chữ cái."""
    source = "".join(_HAN[(i * 3 + k) % len(_HAN)] for k in range(8))
    return i, source, f"Đây là câu dịch bình thường khá dài số {i}."


def _job(bad: dict[int, tuple[str, str]], total: int = 30) -> list[tuple[int, str, str]]:
    pairs = [_normal_line(i) for i in range(1, total + 1)]
    return [(i, *bad[i]) if i in bad else (i, s, t) for i, s, t in pairs]


BAD_REAL = {
    10: ("师父师姐", "Sư tỷ..."),
    20: ("就把这个臭小子", "kia."),
}


class TestPatchTargets:
    def test_real_cases_are_flagged(self) -> None:
        targets = patch_targets(_job(BAD_REAL))
        assert {t.index for t in targets} == {10, 20}

    def test_clean_job_has_no_targets(self) -> None:
        assert patch_targets(_job({})) == []

    def test_limit(self) -> None:
        bad = {i: ("师父师姐不在", "Ừ.") for i in range(1, 60, 1)}
        pairs = [
            (i, "".join(_HAN[(i * 5 + k) % len(_HAN)] for k in range(9)), "Ừ.")
            for i in range(1, 60)
        ] + [_normal_line(i) for i in range(100, 200)]
        assert len(patch_targets(pairs, limit=7)) <= 7
        assert MAX_PATCH_LINES >= 20
        assert bad  # dữ liệu mẫu dùng để đọc hiểu


class TestPatchWindow:
    def test_adds_neighbours_on_both_sides(self) -> None:
        pairs = _job(BAD_REAL)
        targets = [t for t in patch_targets(pairs) if t.index == 10]
        window = patch_window(targets, [i for i, _, _ in pairs], radius=2)
        assert window == [8, 9, 10, 11, 12]

    def test_edges_and_overlap(self) -> None:
        pairs = _job({1: ("师父师姐不在", "Ừ."), 3: ("就把这个臭小子", "kia.")})
        targets = patch_targets(pairs)
        window = patch_window(targets, [i for i, _, _ in pairs], radius=2)
        assert window == [1, 2, 3, 4, 5]


class TestAcceptPatches:
    def _case(self, retried: dict[int, str]) -> dict[int, str]:
        pairs = _job(BAD_REAL)
        return accept_patches(pairs, patch_targets(pairs), retried)

    def test_accepts_fuller_translation(self) -> None:
        got = self._case({10: "Sư phụ, sư tỷ...", 20: "Lôi cái thằng nhóc thối này"})
        assert got == {10: "Sư phụ, sư tỷ...", 20: "Lôi cái thằng nhóc thối này"}

    def test_each_line_judged_separately(self) -> None:
        got = self._case({10: "Sư phụ, sư tỷ...", 20: "kia."})
        assert got == {10: "Sư phụ, sư tỷ..."}

    @pytest.mark.parametrize("bad", ["", "   ", "Sư tỷ...", "Sư 师父 tỷ...", "Sư", "Sư phụ."])
    def test_rejects_unusable_candidate(self, bad: str) -> None:
        assert self._case({10: bad}) == {}

    def test_rejects_candidate_that_copies_neighbour(self) -> None:
        # dòng 11 đã là "Đây là câu dịch bình thường khá dài số 11." → bản vá y hệt = lặp dòng kề
        assert self._case({10: "Đây là câu dịch bình thường khá dài số 11."}) == {}

    def test_rejects_candidate_that_echoes_neighbour(self) -> None:
        echoed = "Sư tỷ, đây là câu dịch bình thường số 11."
        assert self._case({10: echoed}) == {}

    def test_missing_retry_keeps_line(self) -> None:
        assert self._case({}) == {}


class _Translator:
    def __init__(self, answers: dict[int, str] | None = None, error: Exception | None = None):
        self.answers = answers or {}
        self.error = error
        self.calls: list[list[int]] = []

    def has_any_daily_quota(self, _model: str) -> bool:
        return True

    def translate_stage(self, *, input_lines: list[TranslationLine], **_kw: Any):
        self.calls.append([line.index for line in input_lines])
        if self.error is not None:
            raise self.error
        return [
            TranslationLine(index=line.index, start_ms=line.start_ms, end_ms=line.end_ms,
                            text=self.answers.get(line.index, "Đã dịch lại hoàn toàn khác " + str(line.index)),
                            original_text=line.original_text)
            for line in input_lines
        ]


def _request() -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[TranslationStageConfig(kind=TranslationStageKind.LITERAL,
                                       model_name="gemini-3.1-flash-lite")],
        context=TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    )


def _lines(pairs: list[tuple[int, str, str]]) -> tuple[list[TranslationLine], list[TranslationLine]]:
    current = [TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=t,
                               original_text=s) for i, s, t in pairs]
    source = [TranslationLine(index=i, start_ms=i * 1000, end_ms=i * 1000 + 900, text=s,
                              original_text=s) for i, s, _ in pairs]
    return current, source


@pytest.fixture(autouse=True)
def _fresh_log() -> None:
    get_process_log().clear()


class TestUseCaseStep:
    def test_patches_only_flagged_lines_and_leaves_neighbours(self) -> None:
        current, source = _lines(_job(BAD_REAL))
        translator = _Translator({10: "Sư phụ, sư tỷ...", 20: "Lôi cái thằng nhóc thối này"})
        use_case = TranslateSubtitlesUseCase(translator=translator)
        result = use_case._patch_under_translated(current, source, _request(), None)
        before = {line.index: line.text for line in current}
        after = {line.index: line.text for line in result}
        assert {i for i in after if after[i] != before[i]} == {10, 20}
        assert after[10] == "Sư phụ, sư tỷ..."
        # một lượt gọi duy nhất, chỉ gồm dòng cần vá + dòng kề (không phải cả phim)
        assert len(translator.calls) == 1
        assert set(translator.calls[0]) == {8, 9, 10, 11, 12, 18, 19, 20, 21, 22}
        assert find_under_translated([(i, s, after[i]) for i, s, _ in _job(BAD_REAL)]) == []

    def test_neighbours_changed_by_model_are_ignored(self) -> None:
        current, source = _lines(_job(BAD_REAL))
        translator = _Translator({10: "Sư phụ, sư tỷ..."})   # còn lại: câu "khác hẳn"
        result = TranslateSubtitlesUseCase(translator=translator)._patch_under_translated(
            current, source, _request(), None)
        after = {line.index: line.text for line in result}
        assert after[9] == "Đây là câu dịch bình thường khá dài số 9."
        # dòng 20 cũng được vá (bản dịch lại dài hơn, hợp lệ); dòng kề thì KHÔNG bị đụng tới
        assert after[20] == "Đã dịch lại hoàn toàn khác 20"
        assert after[21] == "Đây là câu dịch bình thường khá dài số 21."
        assert after[19] == "Đây là câu dịch bình thường khá dài số 19."

    def test_no_targets_means_no_api_call(self) -> None:
        current, source = _lines(_job({}))
        translator = _Translator()
        result = TranslateSubtitlesUseCase(translator=translator)._patch_under_translated(
            current, source, _request(), None)
        assert result is current
        assert translator.calls == []

    def test_translator_failure_keeps_everything(self) -> None:
        current, source = _lines(_job(BAD_REAL))
        translator = _Translator(error=SubtitleTranslationError("hết hạn mức"))
        result = TranslateSubtitlesUseCase(translator=translator)._patch_under_translated(
            current, source, _request(), None)
        assert result is current

    def test_unimproved_retry_keeps_old_text_and_is_reported(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        current, source = _lines(_job(BAD_REAL))
        translator = _Translator({10: "Sư tỷ...", 20: "kia."})
        with caplog.at_level("INFO"):
            result = TranslateSubtitlesUseCase(translator=translator)._patch_under_translated(
                current, source, _request(), None)
        assert result is current
        assert any("còn nghi" in record.getMessage() for record in caplog.records)

    def test_records_guard_event(self) -> None:
        current, source = _lines(_job(BAD_REAL))
        TranslateSubtitlesUseCase(
            translator=_Translator({10: "Sư phụ, sư tỷ...", 20: "kia."}),
        )._patch_under_translated(current, source, _request(), None)
        events = [e for e in get_process_log().events_since(0) if e["event"] == "under_translated_patched"]
        assert events, "phải ghi sự kiện cho nhật ký chuẩn đoán"
        assert events[-1]["patched"] == [10]
        assert events[-1]["still_suspect"] == [20]
