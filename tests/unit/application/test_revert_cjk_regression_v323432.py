"""[v3.23.432] Test safety-net chống giai đoạn tinh chỉnh chèn CJK vào câu đã sạch."""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine


def _line(index: int, text: str) -> TranslationLine:
    return TranslationLine(index=index, start_ms=0, end_ms=0, text=text)


def _use_case() -> TranslateSubtitlesUseCase:
    # Method không dùng self._translator → truyền None an toàn cho test.
    return TranslateSubtitlesUseCase(translator=None)  # type: ignore[arg-type]


def test_reverts_inserted_cjk_when_prior_clean() -> None:
    uc = _use_case()
    prior = [_line(276, "mệnh tinh vẫn lạc.")]
    out = [_line(276, "mệnh星 vẫn lạc.")]  # localize chèn 星
    fixed = uc._revert_cjk_regressions(out, prior, "localize")
    assert fixed[0].text == "mệnh tinh vẫn lạc."


def test_idiom_revert_recovered() -> None:
    uc = _use_case()
    prior = [_line(139, "tôi nhất định sẽ phát phấn đồ cường,")]
    out = [_line(139, "tôi nhất định sẽ phát奋图强,")]
    fixed = uc._revert_cjk_regressions(out, prior, "localize")
    assert "奋" not in fixed[0].text
    assert fixed[0].text == prior[0].text


def test_literal_from_source_not_reverted() -> None:
    """Giai đoạn literal: input là nguồn CJK → không lùi nhầm."""
    uc = _use_case()
    prior = [_line(1, "命星陨落")]  # nguồn
    out = [_line(1, "còn 命 sót")]  # literal dịch hụt (còn CJK)
    fixed = uc._revert_cjk_regressions(out, prior, "literal")
    assert fixed[0].text == "còn 命 sót"  # KHÔNG lùi về nguồn


def test_clean_output_untouched() -> None:
    uc = _use_case()
    prior = [_line(5, "Xin chào")]
    out = [_line(5, "Chào bạn")]
    fixed = uc._revert_cjk_regressions(out, prior, "style")
    assert fixed[0].text == "Chào bạn"


def test_missing_index_kept() -> None:
    uc = _use_case()
    prior = [_line(1, "a")]
    out = [_line(2, "有 CJK")]  # index không có trong prior
    fixed = uc._revert_cjk_regressions(out, prior, "style")
    assert fixed[0].text == "有 CJK"  # không có bản trước để lùi → giữ nguyên


# [v3.23.433] Test cleanup pass dọn CJK sót thật
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationStageConfig, TranslationStageKind


class _StubTranslator:
    """Translator giả: dịch lại trả bản do caller cấu hình."""
    def __init__(self, retry_map):
        self._retry = retry_map
    def translate_stage(self, *, stage, context, source_lines, input_lines,
                        has_prior_translation, progress_cb, cancel_cb,
                        video_refs, attach_video,
        **_extra,
    ):
        return [_line(ln.index, self._retry.get(ln.index, ln.text)) for ln in source_lines]


def _make_request(source_texts):
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
    )
    import inspect
    # Dung SimpleNamespace-like: chi can .stages/.context/.video_refs/.events
    from types import SimpleNamespace
    stage = TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
    return SimpleNamespace(
        stages=[stage], context=None, video_refs=[], events=[],
    )


def test_cleanup_replaces_when_retry_clean() -> None:
    retry = {18: "Đồ hỗn trướng!"}  # dịch lại SẠCH
    uc = TranslateSubtitlesUseCase(translator=_StubTranslator(retry))
    current = [_line(18, "Đồ hỗn账!")]
    source = [_line(18, "混账")]
    req = _make_request(source)
    out = uc._cleanup_residual_cjk(current, source, req, None)
    assert out[0].text == "Đồ hỗn trướng!"  # đã thay bản sạch


def test_cleanup_keeps_when_retry_still_cjk() -> None:
    retry = {18: "Đồ hỗn账!"}  # dịch lại VẪN sót → không thay
    uc = TranslateSubtitlesUseCase(translator=_StubTranslator(retry))
    current = [_line(18, "Đồ hỗn账!")]
    source = [_line(18, "混账")]
    req = _make_request(source)
    out = uc._cleanup_residual_cjk(current, source, req, None)
    assert out[0].text == "Đồ hỗn账!"  # giữ nguyên (không làm tệ hơn)


def test_cleanup_noop_when_clean() -> None:
    uc = TranslateSubtitlesUseCase(translator=_StubTranslator({}))
    current = [_line(1, "Xin chào")]
    req = _make_request([_line(1, "你好")])
    out = uc._cleanup_residual_cjk(current, [_line(1, "你好")], req, None)
    assert out[0].text == "Xin chào"


def test_cleanup_translator_error_keeps_original() -> None:
    class _Boom:
        def translate_stage(self, **k): raise RuntimeError("API lỗi")
    uc = TranslateSubtitlesUseCase(translator=_Boom())
    current = [_line(18, "Đồ hỗn账!")]
    source = [_line(18, "混账")]
    req = _make_request(source)
    out = uc._cleanup_residual_cjk(current, source, req, None)
    assert out[0].text == "Đồ hỗn账!"  # fallback an toàn


# [v3.23.436] Test guard chống LÀM-RỖNG
def test_reverts_emptied_line() -> None:
    uc = _use_case()
    prior = [_line(144, "Sao ông già này một chút phản ứng cũng không có?")]
    out = [_line(144, "")]  # stage final làm rỗng
    fixed = uc._revert_cjk_regressions(out, prior, "final")
    assert fixed[0].text == "Sao ông già này một chút phản ứng cũng không có?"


def test_emptied_whitespace_only_reverted() -> None:
    uc = _use_case()
    prior = [_line(5, "Nội dung")]
    out = [_line(5, "   ")]  # chỉ khoảng trắng
    fixed = uc._revert_cjk_regressions(out, prior, "style")
    assert fixed[0].text == "Nội dung"


def test_legit_empty_when_prior_also_empty() -> None:
    uc = _use_case()
    prior = [_line(5, "")]
    out = [_line(5, "")]
    fixed = uc._revert_cjk_regressions(out, prior, "style")
    assert fixed[0].text == ""  # cả hai rỗng → không đụng
