"""[v3.23.585] Gemini QUÁ TẢI (503) không được đốt sạch hạn mức ngày của model ít quota.

Nhật ký thật v3.23.584 (phim "灭门惨案II", phân tích 17 đoạn bằng gemini-3.8-flash, hạn
mức 20 request/ngày, 3 API key):

* Lần phân tích đầu chạy 93 phút, tiêu 60 request của cả ba key, xong 1/17 đoạn — mỗi lời
  gọi được thử tới ~11 lượt, mà mỗi lượt 503 vẫn bị server tính vào hạn mức ngày.
* Ứng dụng hoàn lại bộ đếm sau mỗi 503 nên tưởng key còn nguyên: bảy lần server báo hết
  hạn mức thì ứng dụng mới đếm 0, 0, 1, 8, 5, 8, 4 request.
* Lỗi hết hạn mức ngày rơi đúng lượt thử cuối → bỏ cả đoạn dù còn key khác (4 lần).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationLine,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
    QuotaExhaustedError,
    RateLimit,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    OVERLOAD_RETRY_BUDGET_S,
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.overload_retry_policy import (
    overload_attempt_cap,
    scarce_overload_delay,
)
from tests.unit.infrastructure.gemini_call_harness import (
    OK,
    REAL_503,
    REAL_DAILY_429,
    SCARCE_MODEL,
    FakeClock,
    ScriptedGemini,
    bare_analysis_adapter,
    build_translator,
    call_gemini,
    confirmed_scarce_manager,
    daily_requests_used,
    install_fake_clock,
    quota_manager,
)

ABUNDANT_MODEL = "gemini-3.5-flash-lite"

# Bước của kịch bản server giả: nguyên văn hai thông báo lỗi chép từ nhật ký thật.
OVERLOAD = REAL_503
DAILY = REAL_DAILY_429


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


class TestPolicy:
    def test_scarce_model_gets_five_attempts(self) -> None:
        assert overload_attempt_cap(20) == 5
        assert overload_attempt_cap(100) == 5

    def test_abundant_or_unknown_limit_is_not_capped(self) -> None:
        assert overload_attempt_cap(500) is None
        assert overload_attempt_cap(0) is None

    def test_waits_get_longer(self) -> None:
        assert [scarce_overload_delay(n) for n in (1, 2, 3, 4, 5)] == [
            30.0, 60.0, 120.0, 120.0, 120.0
        ]

    def test_five_attempts_cover_the_outages_seen_in_real_logs(self) -> None:
        """Bốn lần chờ phải phủ khoảng thời gian 7 lượt thử kiểu cũ từng phủ (~300s)."""
        old_waits = [30.0, 30.0, 30.0, 42.0, 82.0, 90.0]  # chép từ nhật ký thật
        assert sum(scarce_overload_delay(n) for n in (1, 2, 3, 4)) >= sum(old_waits)


class TestQuotaCounter:
    def test_overloaded_call_stays_in_the_daily_count(self, tmp_path: Path) -> None:
        manager = quota_manager(tmp_path)
        reservation = manager.acquire(SCARCE_MODEL, 1000, key_id="k")
        manager.release(reservation, request_counted=True)
        remaining = manager.get_remaining(SCARCE_MODEL, key_id="k")
        assert remaining["rpd_used"] == 1
        assert remaining["rpm_used"] == 1
        assert remaining["tpm_used"] == 0  # model chưa xử lý gì → hoàn token

    def test_other_failures_are_still_refunded(self, tmp_path: Path) -> None:
        manager = quota_manager(tmp_path)
        manager.release(manager.acquire(SCARCE_MODEL, 1000, key_id="k"))
        remaining = manager.get_remaining(SCARCE_MODEL, key_id="k")
        assert (remaining["rpd_used"], remaining["rpm_used"]) == (0, 0)

    def test_guess_by_model_name_is_not_a_confirmed_limit(self, tmp_path: Path) -> None:
        assert quota_manager(tmp_path).is_daily_limit_confirmed(SCARCE_MODEL) is False

    def test_user_configured_limit_is_confirmed(self, tmp_path: Path) -> None:
        manager = quota_manager(tmp_path, {SCARCE_MODEL: RateLimit(5, 250_000, 20)})
        assert manager.is_daily_limit_confirmed("Gemini-3.8-Flash") is True

    def test_server_stating_the_guessed_number_confirms_it(self, tmp_path: Path) -> None:
        """Ca thật: server nêu 20, trùng bảng suy theo tên — trước đây không ghi gì."""
        manager = quota_manager(tmp_path)
        assert manager.learn_daily_limit_from_server(SCARCE_MODEL, 20) is False
        assert manager.is_daily_limit_confirmed(SCARCE_MODEL) is True
        # Lưu qua các phiên: mở lại ứng dụng vẫn biết đó là số thật.
        assert quota_manager(tmp_path).is_daily_limit_confirmed(SCARCE_MODEL) is True


class TestCallGeminiUnderOverload:
    def test_scarce_model_stops_after_five_attempts(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERLOAD])
        translator, slept = build_translator(monkeypatch, clock, manager, server)
        with pytest.raises(SubtitleTranslationError, match="503 UNAVAILABLE"):
            call_gemini(translator, SCARCE_MODEL)
        assert server.calls == 5  # trước đây ~11 lượt cho MỘT lời gọi
        assert slept == [30.0, 60.0, 120.0, 120.0]
        assert daily_requests_used(manager, SCARCE_MODEL) == 5  # trước đây ứng dụng đếm 0

    def test_nearly_empty_key_is_not_squeezed_further(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Ca thật sáng hôm sau: đoạn 4 thành công ở lượt thử thứ 5 khi key còn 12/20."""
        manager = confirmed_scarce_manager(tmp_path)
        key_id = GeminiQuotaManager.key_fingerprint("KEY_A")
        for _ in range(8):
            manager.acquire(SCARCE_MODEL, 1, key_id=key_id)
            clock.sleep(61.0)
        server = ScriptedGemini([OVERLOAD, OVERLOAD, OVERLOAD, OVERLOAD, OK])
        translator, _slept = build_translator(monkeypatch, clock, manager, server)
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert daily_requests_used(manager, SCARCE_MODEL) == 13

    def test_short_outage_still_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERLOAD, OVERLOAD, OVERLOAD, OVERLOAD, OK])
        translator, _slept = build_translator(monkeypatch, clock, manager, server)
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert daily_requests_used(manager, SCARCE_MODEL) == 5

    def test_abundant_model_keeps_the_time_budget(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = quota_manager(tmp_path, {ABUNDANT_MODEL: RateLimit(15, 250_000, 500)})
        server = ScriptedGemini([OVERLOAD] * 8 + [OK])
        translator, slept = build_translator(monkeypatch, clock, manager, server)
        assert call_gemini(translator, ABUNDANT_MODEL) == {"subtitles": []}
        assert server.calls == 9
        assert sum(slept) <= OVERLOAD_RETRY_BUDGET_S + 120.0

    def test_unconfirmed_guess_keeps_the_time_budget(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Bảng suy theo tên cố ý bảo thủ (có model đoán 20 mà thật 500) → không siết."""
        manager = quota_manager(tmp_path)
        server = ScriptedGemini([OVERLOAD] * 8 + [OK])
        translator, _slept = build_translator(monkeypatch, clock, manager, server)
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert server.calls == 9
        assert daily_requests_used(manager, SCARCE_MODEL) == 9  # 503 vẫn được đếm


class TestDailyQuotaRotation:
    def test_daily_quota_on_the_last_attempt_still_rotates_keys(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Ca thật đoạn 2: các lượt trước tiêu vào 503, lượt cuối nhận 429 hết hạn mức."""
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERLOAD, OVERLOAD, DAILY, OK])
        translator, _slept = build_translator(
            monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B", retry_count=3
        )
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert translator._api_key == "KEY_B"
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_B") == 1

    def test_new_key_gets_its_own_attempts(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERLOAD, DAILY, OVERLOAD, OVERLOAD, OK])
        translator, slept = build_translator(
            monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B", retry_count=2
        )
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        # Quá tải là của model chứ không của key: các lần chờ tiếp tục thưa dần.
        assert slept == [30.0, 60.0, 120.0]
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_B") == 3

    def test_overload_cap_spans_the_key_switch(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Đổi key không cho thêm 5 lượt nữa: một lời gọi vẫn chỉ chịu 5 lần 503."""
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([OVERLOAD, OVERLOAD, DAILY, OVERLOAD])
        translator, _slept = build_translator(
            monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B"
        )
        with pytest.raises(SubtitleTranslationError, match="503 UNAVAILABLE"):
            call_gemini(translator, SCARCE_MODEL)
        assert server.calls == 6  # 5 lần 503 + 1 lần 429 hết hạn mức ngày
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_B") == 3

    def test_daily_quota_with_no_other_key_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        server = ScriptedGemini([DAILY])
        translator, _slept = build_translator(
            monkeypatch, clock, manager, server, retry_count=1
        )
        # [v3.23.586] Báo đúng loại lỗi "hết hạn mức ngày" thay vì lỗi dịch chung chung.
        with pytest.raises(QuotaExhaustedError, match="hết hạn mức request/ngày"):
            call_gemini(translator, SCARCE_MODEL)
        assert server.calls == 1


class _Clip:
    """Một đoạn video đã tải lên Gemini (file gắn với API key đã tải nó)."""

    def __init__(self, remote_name: str) -> None:
        self.remote_name = remote_name
        self.start_sec = 0.0
        self.end_sec = 340.0


class _FileBoundServer:
    """Gemini giả: từ chối (403) file do key khác tải lên, như server thật."""

    def __init__(self) -> None:
        self.sent: list[list] = []

    def generate_content(self, model: str, contents: list, config: object) -> object:
        self.sent.append(list(contents))
        if "files/key_a" in contents:
            raise RuntimeError(
                "403 PERMISSION_DENIED. You do not have permission to access the File "
                "key_a or it may not exist."
            )
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


class TestKeySwitchWhileReservingQuota:
    def test_video_is_reuploaded_before_the_call_not_after_a_403(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Key hiện tại đã bị server chặn → xoay key ngay ở bước đặt chỗ. Trước đây lời
        gọi vẫn mang file của key cũ: 403, mất một request của key mới và một lượt thử."""
        manager = confirmed_scarce_manager(tmp_path)
        manager.mark_daily_exhausted(
            SCARCE_MODEL, key_id=GeminiQuotaManager.key_fingerprint("KEY_A")
        )
        server = _FileBoundServer()
        translator, _slept = build_translator(
            monkeypatch, clock, manager, server, keys="KEY_A\nKEY_B"
        )
        monkeypatch.setattr(
            translator,
            "_resolve_video_handles",
            lambda refs: [ref.remote_name for ref in refs],
        )
        translator.set_video_reupload_callback(
            lambda _api_key, _ranges: [_Clip("files/key_b")]
        )
        config = SimpleNamespace(
            temperature=0.2, response_mime_type="application/json",
            response_schema={}, system_instruction="x",
        )
        data = translator._call_gemini(
            SCARCE_MODEL, "prompt", config, lambda _data: None,
            video_files=["files/key_a"], video_refs=[_Clip("files/key_a")],
        )
        assert data == {"subtitles": []}
        assert server.sent == [["files/key_b", "prompt"]]
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_B") == 1


class _Ref:
    def __init__(self, idx: int) -> None:
        self.chunk_index = idx
        self.remote_name = f"files/f{idx}"
        self.start_sec = idx * 100.0
        self.end_sec = (idx + 1) * 100.0
        self.state = "ACTIVE"


class _Segments:
    """Thay ``_call_gemini`` khi phân tích: các đoạn trong ``overloaded`` luôn quá tải."""

    def __init__(self, overloaded: frozenset[int]) -> None:
        self.overloaded = overloaded
        self.calls: list[int] = []

    def __call__(self, _model, _prompt, _config, _validator, video_refs=None, **_kw):
        segment = int(video_refs[0].chunk_index) if video_refs else -1
        self.calls.append(segment)
        if segment in self.overloaded or (segment == -1 and self.overloaded):
            raise SubtitleTranslationError(f"Gọi Gemini thất bại: {REAL_503}")
        return {
            "source_lang": "zh",
            "characters": f"人物{segment} (Nhân vật {segment}) — vai",
            "overview": f"đoạn {segment + 1}",
            "glossary": "",
        }


def _analysis_adapter(
    manager: GeminiQuotaManager | None, checkpoint_dir: Path | None = None
) -> GeminiSubtitleTranslator:
    return bare_analysis_adapter(manager, checkpoint_dir)


def _analyze(adapter: GeminiSubtitleTranslator, gemini: _Segments, count: int = 4):
    refs = [_Ref(i) for i in range(count)]
    lines = [
        TranslationLine(index=i, start_ms=0, end_ms=1, text=f"c{i}") for i in range(1, 6)
    ]
    with (
        patch.object(
            adapter, "_resolve_video_handles", lambda r: [f"h{i}" for i in range(count)]
        ),
        patch.object(adapter, "_build_config", lambda *a, **k: MagicMock()),
        patch.object(adapter, "_call_gemini", gemini),
        patch.object(adapter, "_ensure_available", lambda: None),
    ):
        return adapter.analyze_global_context(
            lines, "vi", video_refs=refs, model_name=SCARCE_MODEL
        )


class TestAnalysisStopsBurningQuota:
    def test_overload_on_the_first_segment_stops_everything(
        self, tmp_path: Path
    ) -> None:
        """Ca thật lần phân tích đầu: 60 request cho 1/17 đoạn. Nay dừng ở đoạn đầu."""
        gemini = _Segments(overloaded=frozenset({0, 1, 2, 3}))
        with pytest.raises(SubtitleTranslationError, match="quá tải"):
            _analyze(_analysis_adapter(confirmed_scarce_manager(tmp_path)), gemini)
        # Không gửi đoạn 2-4, không thử lại cuối phiên, không gọi nốt lượt "chỉ văn bản".
        assert gemini.calls == [0]

    def test_overload_mid_run_keeps_finished_segments(self, tmp_path: Path) -> None:
        gemini = _Segments(overloaded=frozenset({2}))
        checkpoints = tmp_path / "ckpt"
        result = _analyze(
            _analysis_adapter(confirmed_scarce_manager(tmp_path), checkpoints), gemini
        )
        assert gemini.calls == [0, 1, 2]
        assert "đoạn 2" in result.overview
        # Tiến độ còn nguyên → lần chạy sau chỉ gửi hai đoạn còn thiếu.
        again = _Segments(overloaded=frozenset())
        manager = confirmed_scarce_manager(tmp_path)
        _analyze(_analysis_adapter(manager, checkpoints), again)
        assert again.calls == [2, 3]

    def test_abundant_model_still_tries_every_segment(self, tmp_path: Path) -> None:
        manager = quota_manager(tmp_path, {SCARCE_MODEL: RateLimit(15, 250_000, 500)})
        gemini = _Segments(overloaded=frozenset({1}))
        result = _analyze(_analysis_adapter(manager), gemini)
        assert gemini.calls == [0, 1, 2, 3, 1]  # đi hết rồi thử lại đoạn hỏng
        assert "đoạn 4" in result.overview

    def test_without_quota_manager_behaviour_is_unchanged(self) -> None:
        gemini = _Segments(overloaded=frozenset({1}))
        _analyze(_analysis_adapter(None), gemini)
        assert gemini.calls == [0, 1, 2, 3, 1]


class TestIncompleteAnalysisIsMarked:
    """Kết quả thiếu đoạn phải tự mang thông tin đó tới giao diện."""

    def test_overload_stop_is_carried_in_the_result(self, tmp_path: Path) -> None:
        gemini = _Segments(overloaded=frozenset({2}))
        result = _analyze(_analysis_adapter(confirmed_scarce_manager(tmp_path)), gemini)
        assert result.missing_segments == (3, 4)
        assert result.total_segments == 4
        assert result.stop_reason == "overload"

    def test_complete_analysis_is_not_marked(self, tmp_path: Path) -> None:
        gemini = _Segments(overloaded=frozenset())
        result = _analyze(_analysis_adapter(confirmed_scarce_manager(tmp_path)), gemini)
        assert result.missing_segments == ()
        assert result.stop_reason == ""
