"""[v3.23.592] Hồi quy: quá tải kéo dài mà ứng dụng KHÔNG thử API key khác.

Nhật ký thật v3.23.591 (ba API key, phim "灭门惨案II"):

    08:33:19 Server báo HẾT HẠN MỨC NGÀY cho 'gemini-3.8-flash' (key …0d53). …
    08:33:19 Xoay API key #1 → #2 (còn 20 req/ngày) … [nguồn: SERVER từ chối].
    08:33:25 Lỗi tạm thời từ Gemini (lần 1/3): 503 UNAVAILABLE. Chờ 30.0s …
    08:33:58 … 08:35:10 … 08:38:16 …                       ← 5 lượt 503 liền trên key #2
    08:40:51 Gemini QUÁ TẢI: dừng sau 5 lượt thử để GIỮ hạn mức ngày …
    08:40:51 Gemini QUÁ TẢI kéo dài ở đoạn 12 — DỪNG gửi các đoạn còn lại …
    08:41:20 Chọn API key #3 (còn 20 req/ngày) …            ← người dùng bấm phân tích lại
    08:42:31 Phân tích ngữ cảnh TUẦN TỰ đoạn 12/17 …
    08:43:17 Phân tích ngữ cảnh TUẦN TỰ đoạn 13/17 …        ← key #3: xong ngay lượt đầu

    09:12:03 Lỗi tạm thời từ Gemini (lần 1/5): 503 UNAVAILABLE. …  (gemini-3.5-flash-lite)
    …                                                       ← 11 lượt 503 trên key #1
    09:26:17 Dịch phụ đề thất bại: Gọi Gemini thất bại: 503 UNAVAILABLE. …

Ở cả hai lần, các key còn lại không được thử lần nào.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ModelOverloadedError,
    SubtitleTranslationError,
)
from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    OVERLOAD_RETRY_BUDGET_S,
    GeminiSubtitleTranslator,
)
from subtitles_extractor.infrastructure.translation.overload_retry_policy import (
    PLENTIFUL_ROTATE_AFTER,
    SCARCE_MAX_ATTEMPTS,
    SCARCE_PROBE_ATTEMPTS,
    overload_key_cap,
    should_try_other_key,
)
from tests.unit.infrastructure.gemini_call_harness import (
    REAL_503,
    SCARCE_MODEL,
    FakeClock,
    build_translator,
    call_gemini,
    confirmed_scarce_manager,
    daily_requests_used,
    install_fake_clock,
)

ABUNDANT_MODEL = "gemini-3.5-flash-lite"
THREE_KEYS = "KEY_A\nKEY_B\nKEY_C"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    return install_fake_clock(monkeypatch)


class _KeyedServer:
    """Gemini giả: mỗi API key quá tải hay không là chuyện riêng của key đó."""

    def __init__(self, overloaded: set[str]) -> None:
        self.overloaded = overloaded
        self.translator: GeminiSubtitleTranslator | None = None
        self.calls: list[str] = []
        self.sent: list[list[Any]] = []

    def generate_content(self, model: str, contents: list, config: object) -> object:
        assert self.translator is not None
        key = self.translator._api_key
        self.calls.append(key)
        self.sent.append(list(contents))
        if key in self.overloaded:
            raise RuntimeError(REAL_503)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)

    def count(self, key: str) -> int:
        return self.calls.count(key)


def _translator(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    manager: GeminiQuotaManager | None,
    overloaded: set[str],
    *,
    keys: str = THREE_KEYS,
    retry_count: int = 3,
) -> tuple[GeminiSubtitleTranslator, _KeyedServer, list[float]]:
    server = _KeyedServer(overloaded)
    translator, slept = build_translator(
        monkeypatch, clock, manager, server, keys=keys, retry_count=retry_count
    )
    server.translator = translator
    return translator, server, slept


class TestPolicy:
    def test_first_key_keeps_its_full_cap(self) -> None:
        assert overload_key_cap(SCARCE_MAX_ATTEMPTS, 0) == 5

    def test_later_keys_are_only_probed(self) -> None:
        assert SCARCE_PROBE_ATTEMPTS == 2
        assert overload_key_cap(SCARCE_MAX_ATTEMPTS, 1) == 2
        assert overload_key_cap(SCARCE_MAX_ATTEMPTS, 2) == 2

    def test_probe_never_exceeds_a_smaller_cap(self) -> None:
        assert overload_key_cap(1, 1) == 1

    def test_abundant_model_has_no_cap(self) -> None:
        assert overload_key_cap(None, 0) is None
        assert overload_key_cap(None, 2) is None

    def test_scarce_key_moves_on_at_its_cap(self) -> None:
        assert not should_try_other_key(5, 4)
        assert should_try_other_key(5, 5)
        assert should_try_other_key(2, 2)

    def test_abundant_key_moves_on_after_three_in_a_row(self) -> None:
        assert PLENTIFUL_ROTATE_AFTER == 3
        assert not should_try_other_key(None, 2)
        assert should_try_other_key(None, 3)


class TestScarceModelTriesOtherKeys:
    def test_real_case_analysis_continues_on_the_next_key(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Trước bản này: 5 lượt 503 trên một key rồi DỪNG, key còn lại không được thử."""
        manager = confirmed_scarce_manager(tmp_path)
        translator, server, slept = _translator(monkeypatch, clock, manager, {"KEY_A"})
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A"] * 5 + ["KEY_B"]
        assert translator._api_key == "KEY_B"
        # Bốn lần chờ trên key đầu; đổi key thì gửi lại NGAY.
        assert slept == [30.0, 60.0, 120.0, 120.0]

    def test_overloaded_calls_are_counted_per_key(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        translator, _server, _slept = _translator(monkeypatch, clock, manager, {"KEY_A"})
        call_gemini(translator, SCARCE_MODEL)
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_A") == 5
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_B") == 1
        assert daily_requests_used(manager, SCARCE_MODEL, "KEY_C") == 0

    def test_other_keys_are_only_probed_twice_each(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Mọi key đều quá tải: 5 + 2 + 2 lượt rồi dừng — không đốt 5 lượt mỗi key."""
        manager = confirmed_scarce_manager(tmp_path)
        translator, server, slept = _translator(
            monkeypatch, clock, manager, {"KEY_A", "KEY_B", "KEY_C"}
        )
        with pytest.raises(SubtitleTranslationError, match="503"):
            call_gemini(translator, SCARCE_MODEL)
        assert server.calls == ["KEY_A"] * 5 + ["KEY_B"] * 2 + ["KEY_C"] * 2
        assert slept == [30.0, 60.0, 120.0, 120.0, 30.0, 30.0]
        assert [
            daily_requests_used(manager, SCARCE_MODEL, key)
            for key in ("KEY_A", "KEY_B", "KEY_C")
        ] == [5, 2, 2]

    def test_second_probe_key_can_save_the_call(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        translator, server, _slept = _translator(
            monkeypatch, clock, manager, {"KEY_A", "KEY_B"}
        )
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A"] * 5 + ["KEY_B"] * 2 + ["KEY_C"]

    def test_key_without_daily_quota_is_not_probed(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """Đúng ca thật: key #1 đã hết hạn mức ngày nên chỉ còn key #3 để thử."""
        manager = confirmed_scarce_manager(tmp_path)
        manager.mark_daily_exhausted(
            SCARCE_MODEL, key_id=GeminiQuotaManager.key_fingerprint("KEY_A")
        )
        translator, server, _slept = _translator(monkeypatch, clock, manager, {"KEY_B"})
        translator._switch_to_key(1)
        assert call_gemini(translator, SCARCE_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_B"] * 5 + ["KEY_C"]

    def test_single_key_behaves_as_before(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        translator, server, slept = _translator(
            monkeypatch, clock, manager, {"KEY_A"}, keys="KEY_A"
        )
        with pytest.raises(SubtitleTranslationError, match="503"):
            call_gemini(translator, SCARCE_MODEL)
        assert server.calls == ["KEY_A"] * 5
        assert slept == [30.0, 60.0, 120.0, 120.0]

    def test_log_names_the_key_change(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clock: FakeClock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        translator, _server, _slept = _translator(monkeypatch, clock, manager, {"KEY_A"})
        with caplog.at_level("WARNING"):
            call_gemini(translator, SCARCE_MODEL)
        assert "QUÁ TẢI kéo dài trên key #1 → thử key #2" in caplog.text

    def test_final_stop_says_how_many_keys_were_tried(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clock: FakeClock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        manager = confirmed_scarce_manager(tmp_path)
        translator, _server, _slept = _translator(
            monkeypatch, clock, manager, {"KEY_A", "KEY_B", "KEY_C"}
        )
        with caplog.at_level("WARNING"), pytest.raises(SubtitleTranslationError):
            call_gemini(translator, SCARCE_MODEL)
        assert "dừng sau 9 lượt thử trên 3 key" in caplog.text


class TestAbundantModelTriesOtherKeys:
    def test_real_case_translation_moves_to_the_next_key(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        """Trước bản này: 11 lượt 503 trên key #1 rồi cả lượt dịch thất bại."""
        translator, server, slept = _translator(
            monkeypatch, clock, None, {"KEY_A"}, retry_count=5
        )
        assert call_gemini(translator, ABUNDANT_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A"] * 3 + ["KEY_B"]
        assert translator._api_key == "KEY_B"
        assert len(slept) == 2  # chờ sau lượt 1 và 2; lượt 3 đổi key, gửi ngay

    def test_every_key_gets_its_turn_before_giving_up(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        translator, server, slept = _translator(
            monkeypatch, clock, None, {"KEY_A", "KEY_B", "KEY_C"}, retry_count=5
        )
        with pytest.raises(SubtitleTranslationError, match="503"):
            call_gemini(translator, ABUNDANT_MODEL)
        assert server.calls[:9] == ["KEY_A"] * 3 + ["KEY_B"] * 3 + ["KEY_C"] * 3
        # Hết key để thử thì ở lại key cuối cho tới khi hết ngân sách chờ.
        assert set(server.calls[9:]) == {"KEY_C"}
        assert sum(slept) <= OVERLOAD_RETRY_BUDGET_S + 120.0

    def test_third_key_can_save_the_call(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        translator, server, _slept = _translator(
            monkeypatch, clock, None, {"KEY_A", "KEY_B"}, retry_count=5
        )
        assert call_gemini(translator, ABUNDANT_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A"] * 3 + ["KEY_B"] * 3 + ["KEY_C"]

    def test_short_overload_does_not_change_key(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        """Hai lượt 503 rồi xong: ở lại key cũ (không phải tải lại video)."""
        translator, server, _slept = _translator(
            monkeypatch, clock, None, set(), retry_count=5
        )
        script = iter([REAL_503, REAL_503])

        def _flaky(model: str, contents: list, config: object) -> object:
            server.calls.append(translator._api_key)
            step = next(script, None)
            if step is not None:
                raise RuntimeError(step)
            return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)

        monkeypatch.setattr(server, "generate_content", _flaky)
        assert call_gemini(translator, ABUNDANT_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A"] * 3
        assert translator._api_key == "KEY_A"

    def test_single_key_behaves_as_before(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        translator, server, _slept = _translator(
            monkeypatch, clock, None, {"KEY_A"}, keys="KEY_A", retry_count=5
        )
        with pytest.raises(SubtitleTranslationError, match="503"):
            call_gemini(translator, ABUNDANT_MODEL)
        assert set(server.calls) == {"KEY_A"}


class _OtherThreadSwitches(_KeyedServer):
    """Như trên, nhưng trong lúc lượt gửi thứ ``at`` còn đang bay thì một luồng dịch
    song song khác đã chuyển ứng dụng sang key ``to_index``."""

    def __init__(self, overloaded: set[str], *, at: int, to_index: int) -> None:
        super().__init__(overloaded)
        self.at = at
        self.to_index = to_index

    def generate_content(self, model: str, contents: list, config: object) -> object:
        assert self.translator is not None
        key = self.translator._api_key
        self.calls.append(key)
        self.sent.append(list(contents))
        if len(self.calls) == self.at:
            self.translator._switch_to_key(self.to_index)
        if key in self.overloaded:
            raise RuntimeError(REAL_503)
        return SimpleNamespace(text='{"subtitles": []}', usage_metadata=None)


def _parallel_translator(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    overloaded: set[str],
    *,
    at: int,
    to_index: int,
) -> tuple[GeminiSubtitleTranslator, _OtherThreadSwitches]:
    server = _OtherThreadSwitches(overloaded, at=at, to_index=to_index)
    translator, _slept = build_translator(
        monkeypatch, clock, None, server, keys=THREE_KEYS, retry_count=5
    )
    server.translator = translator
    return translator, server


class TestAnotherThreadAlreadySwitchedKey:
    """Dịch song song: nhiều lời gọi cùng dính 503 trên một key, một luồng chuyển key
    trước. Luồng còn lại phải đi theo, không tự chuyển lần nữa."""

    def test_call_follows_the_switch_instead_of_switching_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clock: FakeClock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Lượt 503 thứ ba trên key A còn đang bay thì luồng khác đã sang key B. Chọn
        "key chưa thử, không phải key hiện tại" ở đây sẽ đưa cả ứng dụng QUAY VỀ key A
        đang quá tải."""
        translator, server = _parallel_translator(
            monkeypatch, clock, {"KEY_A"}, at=3, to_index=1
        )
        with caplog.at_level("WARNING"):
            assert call_gemini(translator, ABUNDANT_MODEL) == {"subtitles": []}
        assert server.calls == ["KEY_A"] * 3 + ["KEY_B"]
        assert translator._api_key == "KEY_B"
        assert "→ thử key #" not in caplog.text

    def test_failures_are_counted_for_the_key_that_was_used(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        """Một lượt 503 trên key A, rồi luồng khác sang key B (cũng quá tải): key B
        vẫn được thử đủ ba lượt của nó trước khi đi tiếp."""
        translator, server = _parallel_translator(
            monkeypatch, clock, {"KEY_A", "KEY_B"}, at=1, to_index=1
        )
        assert call_gemini(translator, ABUNDANT_MODEL) == {"subtitles": []}
        assert server.calls[:4] == ["KEY_A"] + ["KEY_B"] * 3
        assert server.calls[-1] == "KEY_C"
        assert server.count("KEY_B") == 3

    def test_video_is_refreshed_when_the_key_changed_under_the_call(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
    ) -> None:
        """Key đổi giữa hai lượt gửi của cùng lời gọi: lượt sau phải mang file của key
        mới, không gửi file của key cũ để rồi nhận 403."""
        translator, server = _parallel_translator(
            monkeypatch, clock, {"KEY_A"}, at=1, to_index=1
        )
        monkeypatch.setattr(
            translator,
            "_resolve_video_handles",
            lambda refs: [ref.remote_name for ref in refs],
        )
        reuploads: list[str] = []

        def _reupload(api_key: str, _ranges: object) -> list[_Clip]:
            reuploads.append(api_key)
            return [_Clip(f"files/{api_key.lower()}")]

        translator.set_video_reupload_callback(_reupload)
        config = SimpleNamespace(
            temperature=0.2, response_mime_type="application/json",
            response_schema={}, system_instruction="x",
        )
        data = translator._call_gemini(
            ABUNDANT_MODEL, "prompt", config, lambda _data: None,
            video_files=["files/key_a"], video_refs=[_Clip("files/key_a")],
        )
        assert data == {"subtitles": []}
        assert server.calls == ["KEY_A", "KEY_B"]
        assert reuploads == ["KEY_B"]
        assert server.sent == [["files/key_a", "prompt"], ["files/key_b", "prompt"]]


class _Clip:
    """Một đoạn video đã tải lên Gemini (file gắn với API key đã tải nó)."""

    def __init__(self, remote_name: str) -> None:
        self.remote_name = remote_name
        self.start_sec = 0.0
        self.end_sec = 340.0


class TestVideoFollowsTheKey:
    def test_overload_with_video_does_not_rotate_so_nothing_is_reuploaded(
        self, monkeypatch: pytest.MonkeyPatch, clock: FakeClock, tmp_path: Path
    ) -> None:
        """[v3.23.631] Quá tải là của MODEL: nhật ký thật 17 lần xoay key vì quá tải chỉ 2 lần
        được trả lời ngay, mà mỗi lần xoay kèm video buộc tải lại đoạn (~40 s + request bị tính
        hạn mức). Lời gọi đính video ở lại key, không tải lại.

        (Trước v3.23.631 test này khẳng định ngược lại: xoay sang KEY_B và tải lại video trước
        khi gửi — hành vi vẫn còn cho lời gọi KHÔNG có video, xem các test phía trên.)
        """
        manager = confirmed_scarce_manager(tmp_path)
        translator, server, _slept = _translator(monkeypatch, clock, manager, {"KEY_A"})
        monkeypatch.setattr(
            translator,
            "_resolve_video_handles",
            lambda refs: [ref.remote_name for ref in refs],
        )
        reuploads: list[str] = []

        def _reupload(api_key: str, _ranges: object) -> list[_Clip]:
            reuploads.append(api_key)
            return [_Clip(f"files/{api_key.lower()}")]

        translator.set_video_reupload_callback(_reupload)
        config = SimpleNamespace(
            temperature=0.2, response_mime_type="application/json",
            response_schema={}, system_instruction="x",
        )
        with pytest.raises(ModelOverloadedError):
            translator._call_gemini(
                SCARCE_MODEL, "prompt", config, lambda _data: None,
                video_files=["files/key_a"], video_refs=[_Clip("files/key_a")],
            )
        assert reuploads == []
        assert server.count("KEY_B") == 0 and server.count("KEY_C") == 0
        assert all(contents == ["files/key_a", "prompt"] for contents in server.sent)
