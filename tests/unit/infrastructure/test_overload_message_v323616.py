"""[v3.23.616] Bỏ cuộc vì Gemini quá tải: báo bằng lời dễ hiểu, gợi ý đổi model.

app_log thật 14/10 20:55, 21:09, 21:35 (phim "天骄女友退婚" tập 1): ba lượt dịch liền thất
bại sau 8-14 phút với nguyên khối JSON tiếng Anh của máy chủ ("Gọi Gemini thất bại: 503
UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high
demand…"). Người dùng tự đổi lần lượt 3.6-flash → 3.5-flash → 3.5-flash-lite; tới model
nhẹ nhất thì dịch xong ngay, phần 60/95 dòng đã dịch bằng 3.8-flash được giữ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ModelOverloadedError,
    SubtitleTranslationError,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.overload_retry_policy import (
    overload_failure_message,
)
from subtitles_extractor.presentation.utils.error_humanizer import humanize_gemini_error
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_503,
    SCARCE_MODEL,
    FakeClock,
    ScriptedGemini,
    build_translator,
    call_gemini,
    confirmed_scarce_manager,
    install_fake_clock,
)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


class TestGiveUp:
    def _fail(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> ModelOverloadedError:
        server = ScriptedGemini([REAL_503])
        translator, _slept = build_translator(
            monkeypatch, clock, confirmed_scarce_manager(tmp_path), server
        )
        with pytest.raises(ModelOverloadedError) as caught:
            call_gemini(translator, SCARCE_MODEL)
        return caught.value

    def test_overload_has_its_own_error_type(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        error = self._fail(monkeypatch, clock, tmp_path)
        assert isinstance(error, SubtitleTranslationError)  # nơi bắt cũ vẫn bắt được
        assert error.model_name == SCARCE_MODEL

    def test_message_is_short_vietnamese_without_server_json(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        message = str(self._fail(monkeypatch, clock, tmp_path))
        assert "{'error'" not in message
        assert "high demand" not in message
        assert SCARCE_MODEL in message
        assert "5 lượt" in message
        assert "503 UNAVAILABLE" in message  # mã gọn để giao diện nhận dạng nhóm lỗi

    def test_wrapped_error_is_still_recognised_as_overload(self) -> None:
        error = ModelOverloadedError("Gemini đang quá tải", model_name=SCARCE_MODEL)
        assert GeminiSubtitleTranslator._is_overload_error(error) is True
        assert GeminiSubtitleTranslator._is_retryable(error) is True


class TestMessage:
    def test_states_attempts_keys_and_minutes(self) -> None:
        message = overload_failure_message(
            "gemini-3.8-flash", attempts=9, keys=3, waited_s=512.0,
            reason="503 UNAVAILABLE",
        )
        assert "9 lượt" in message and "3 API key" in message and "9 phút" in message

    def test_never_says_zero_minutes(self) -> None:
        message = overload_failure_message(
            "m", attempts=1, keys=1, waited_s=3.0, reason="503 UNAVAILABLE"
        )
        assert "1 phút" in message


class TestHumanizer:
    def test_overload_hint_suggests_a_lighter_model(self) -> None:
        raw = overload_failure_message(
            "gemini-3.8-flash", attempts=9, keys=3, waited_s=512.0,
            reason="503 UNAVAILABLE",
        )
        friendly = humanize_gemini_error(raw)
        assert "flash-lite" in friendly
        assert "được giữ" in friendly

    def test_a_number_500_is_not_a_server_error(self) -> None:
        raw = "Không đọc được tệp: phụ đề 1500 dòng vượt giới hạn."
        assert humanize_gemini_error(raw) == raw

    def test_real_500_internal_is_still_recognised(self) -> None:
        raw = "500 INTERNAL. {'error': {'code': 500, 'message': 'Internal error'}}"
        assert "quá tải" in humanize_gemini_error(raw)
