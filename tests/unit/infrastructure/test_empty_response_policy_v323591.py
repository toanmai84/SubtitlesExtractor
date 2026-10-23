"""[v3.23.591] Hồi quy: Gemini trả rỗng giữa lúc quá tải → ứng dụng bỏ luôn video.

Nhật ký thật v3.23.590 (phân tích 17 đoạn bằng gemini-3.8-flash, phim "灭门惨案II"):

    20:05:34 Gemini trả rỗng (candidates=None) khi đính video — thử lại KHÔNG đính video
             (video có thể kích hoạt safety filter).
    20:05:59 Gemini đang QUÁ TẢI: 503 UNAVAILABLE. …
    21:30:00 Gemini trả rỗng (candidates=None) khi đính video — thử lại KHÔNG đính video …
    21:30:43 Lỗi tạm thời từ Gemini (lần 2/3): 503 UNAVAILABLE. …
    04:57:00 Phân tích ngữ cảnh TUẦN TỰ đoạn 6/17 …        ← cùng video, cùng prompt
    04:57:32 Phân tích ngữ cảnh TUẦN TỰ đoạn 7/17 …        ← xong sau 32 giây

Hai lần trả rỗng đều ở đoạn 6, đều giữa các lượt 503; sáng hôm sau cùng đoạn video ấy chạy
bình thường — video không hề bị chặn. Bỏ video ở khâu phân tích là bỏ căn cứ gán người
nói/người nghe của cả đoạn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
)
from subtitles_extractor.infrastructure.translation.empty_response_policy import (
    BLOCKING_REASONS,
    UNEXPLAINED_EMPTY_RETRIES_WITH_VIDEO,
    EmptyResponse,
    diagnose_empty_response,
    should_drop_video,
)
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_503,
    SCARCE_MODEL,
    FakeClock,
    build_translator,
    call_gemini,
)

VIDEO = "files/segment6"
OK = types.GenerateContentResponse(
    candidates=[
        types.Candidate(
            content=types.Content(parts=[types.Part(text='{"subtitles": []}')]),
            finish_reason=types.FinishReason.STOP,
        )
    ]
)
#: Đúng dạng nhật ký thật ghi lại: ``candidates=None``, máy chủ không nêu gì thêm.
UNEXPLAINED_EMPTY = types.GenerateContentResponse()
PROMPT_BLOCKED = types.GenerateContentResponse(
    prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason=types.BlockedReason.PROHIBITED_CONTENT
    )
)
OUTPUT_BLOCKED = types.GenerateContentResponse(
    candidates=[types.Candidate(finish_reason=types.FinishReason.SAFETY)]
)
OUT_OF_TOKENS = types.GenerateContentResponse(
    candidates=[types.Candidate(finish_reason=types.FinishReason.MAX_TOKENS)]
)


class _Server:
    """Gemini giả: trả lần lượt các phản hồi của kịch bản, ghi lại nội dung đã gửi."""

    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.sent: list[list[object]] = []

    def generate_content(self, model: str, contents: list, config: object) -> object:
        step = self.script[min(len(self.sent), len(self.script) - 1)]
        self.sent.append(list(contents))
        if isinstance(step, str):
            raise RuntimeError(step)
        return step

    @property
    def with_video(self) -> list[bool]:
        return [VIDEO in contents for contents in self.sent]


def _call(
    monkeypatch: pytest.MonkeyPatch,
    script: list[object],
    *,
    video: bool = True,
    retry_count: int = 3,
) -> tuple[_Server, list[float], dict | Exception]:
    server = _Server(script)
    translator, slept = build_translator(
        monkeypatch, FakeClock(), None, server, retry_count=retry_count
    )
    kwargs = {"video_files": [VIDEO]} if video else {}
    try:
        outcome: dict | Exception = call_gemini(translator, SCARCE_MODEL, **kwargs)
    except SubtitleTranslationError as exc:
        outcome = exc
    return server, slept, outcome


class TestDiagnosis:
    def test_real_log_shape_has_no_reason(self) -> None:
        diagnosis = diagnose_empty_response(UNEXPLAINED_EMPTY)
        assert diagnosis == EmptyResponse()
        assert not diagnosis.blocked
        assert diagnosis.describe() == "máy chủ không nêu lý do"

    def test_blocked_prompt_is_recognised(self) -> None:
        diagnosis = diagnose_empty_response(PROMPT_BLOCKED)
        assert diagnosis.block_reason == "PROHIBITED_CONTENT"
        assert diagnosis.blocked
        assert diagnosis.describe() == "block_reason=PROHIBITED_CONTENT"

    def test_blocked_output_is_recognised(self) -> None:
        diagnosis = diagnose_empty_response(OUTPUT_BLOCKED)
        assert (diagnosis.block_reason, diagnosis.finish_reason) == ("", "SAFETY")
        assert diagnosis.blocked

    def test_running_out_of_tokens_is_not_a_block(self) -> None:
        diagnosis = diagnose_empty_response(OUT_OF_TOKENS)
        assert diagnosis.finish_reason == "MAX_TOKENS"
        assert not diagnosis.blocked
        assert diagnosis.describe() == "finish_reason=MAX_TOKENS"

    def test_unspecified_reason_counts_as_none(self) -> None:
        response = types.GenerateContentResponse(
            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                block_reason=types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
            )
        )
        assert diagnose_empty_response(response) == EmptyResponse()

    def test_plain_strings_and_missing_fields_are_accepted(self) -> None:
        """Đồ giả trong test khác (SimpleNamespace) không có các trường của SDK."""
        assert diagnose_empty_response(SimpleNamespace(text=None)) == EmptyResponse()
        response = SimpleNamespace(
            prompt_feedback=SimpleNamespace(block_reason="safety"), candidates=[]
        )
        assert diagnose_empty_response(response).block_reason == "SAFETY"

    @pytest.mark.parametrize("reason", sorted(BLOCKING_REASONS))
    def test_every_blocking_finish_reason_blocks(self, reason: str) -> None:
        assert EmptyResponse(finish_reason=reason).blocked

    @pytest.mark.parametrize("reason", ["STOP", "MAX_TOKENS", "OTHER", "LANGUAGE", ""])
    def test_other_finish_reasons_do_not_block(self, reason: str) -> None:
        assert not EmptyResponse(finish_reason=reason).blocked


class TestShouldDropVideo:
    def test_blocked_drops_at_once(self) -> None:
        assert should_drop_video(EmptyResponse(block_reason="SAFETY"), 0)

    def test_first_unexplained_empty_keeps_the_video(self) -> None:
        assert UNEXPLAINED_EMPTY_RETRIES_WITH_VIDEO == 1
        assert not should_drop_video(EmptyResponse(), 1)

    def test_second_unexplained_empty_drops_it(self) -> None:
        assert should_drop_video(EmptyResponse(), 2)


class TestCallKeepsVideoOnUnexplainedEmpty:
    def test_real_case_retry_still_carries_the_video(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Trước bản này lượt thử thứ hai đi KHÔNG video — phân tích đoạn 6 "mù"."""
        server, slept, outcome = _call(monkeypatch, [UNEXPLAINED_EMPTY, OK])
        assert outcome == {"subtitles": []}
        assert server.with_video == [True, True]
        assert len(slept) == 1

    def test_real_case_overload_after_the_empty_keeps_the_video(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Đúng diễn biến 21:30: rỗng, rồi 503, rồi mới xong — mọi lượt đều kèm video."""
        server, _slept, outcome = _call(
            monkeypatch, [UNEXPLAINED_EMPTY, REAL_503, OK]
        )
        assert outcome == {"subtitles": []}
        assert server.with_video == [True, True, True]

    def test_second_empty_falls_back_to_no_video(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, _slept, outcome = _call(
            monkeypatch, [UNEXPLAINED_EMPTY, UNEXPLAINED_EMPTY, OK]
        )
        assert outcome == {"subtitles": []}
        assert server.with_video == [True, True, False]

    def test_blocked_prompt_drops_the_video_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, slept, outcome = _call(monkeypatch, [PROMPT_BLOCKED, OK])
        assert outcome == {"subtitles": []}
        assert server.with_video == [True, False]
        assert slept == []

    def test_blocked_output_drops_the_video_at_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, _slept, outcome = _call(monkeypatch, [OUTPUT_BLOCKED, OK])
        assert outcome == {"subtitles": []}
        assert server.with_video == [True, False]

    def test_always_empty_stops_after_a_bounded_number_of_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lượt thử lại kèm video là MỘT lượt thêm; sau đó vẫn đủ ba lượt không video
        như trước (bỏ video, chờ, bỏ cuộc)."""
        server, _slept, outcome = _call(monkeypatch, [UNEXPLAINED_EMPTY])
        assert isinstance(outcome, SubtitleTranslationError)
        assert "máy chủ không nêu lý do" in str(outcome)
        assert server.with_video == [True, True, False, False]

    def test_call_without_video_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, slept, outcome = _call(monkeypatch, [UNEXPLAINED_EMPTY, OK], video=False)
        assert outcome == {"subtitles": []}
        assert server.with_video == [False, False]
        assert len(slept) == 1


class TestLogSaysWhy:
    def test_unexplained_empty_is_reported_as_transient(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            _call(monkeypatch, [UNEXPLAINED_EMPTY, OK])
        assert "máy chủ không nêu lý do" in caplog.text
        assert "thử lại VẪN KÈM video" in caplog.text
        assert "safety filter" not in caplog.text

    def test_block_reason_is_written_to_the_log(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            _call(monkeypatch, [PROMPT_BLOCKED, OK])
        assert "block_reason=PROHIBITED_CONTENT" in caplog.text
        assert "thử lại KHÔNG đính video" in caplog.text
