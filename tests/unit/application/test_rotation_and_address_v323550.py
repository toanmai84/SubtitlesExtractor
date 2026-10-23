"""[v3.23.550] Hồi quy cho các lỗi tìm thấy trong chuẩn đoán thật (phim 3902 dòng, v3.23.549).

1. Xoay key ở yêu cầu CUỐI của khâu phân tích → tải lại CẢ 22 đoạn video (5 phút 45 giây) chỉ để
   gửi một yêu cầu cần một đoạn.
2. Cache upload không có hạn sống: file Gemini tự xoá sau 48 giờ (và trả 403 chứ không phải 404),
   nên mục cũ nào cũng bị đem ra thử trước.
3. Đồng hồ canh "giao diện kẹt bận" bắn 158ms sau khi phân tích xong — tín hiệu kết thúc hợp lệ
   vẫn đang xếp hàng, không hề bị mất.
4. Phân tích mục thuật ngữ tách ``/`` TRƯỚC khi bỏ ngoặc → "Thổ Cẩu (chó ta/chó cỏ)" không bao giờ
   khớp; thành ngữ dịch tự nhiên bị trộn chung với vi phạm tên riêng.
5. Bảng xưng hô liệt kê nhiều đại từ cho một cặp ("tự xưng 'tôi/Mộ Vân/chị'") → khâu dịch tự chọn.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from subtitles_extractor.application.services.roster_format import ambiguous_address_rules
from subtitles_extractor.application.services.translation_diagnostics import (
    GLOSSARY_KIND_EXPRESSION,
    GLOSSARY_KIND_NAME,
    detect_quality_flags,
    glossary_alternatives,
    glossary_entry_kind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    _accepts_ranges_argument,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    GeminiVideoContextProvider,
    is_upload_expired,
)


# ── 1. Tải lại CHỈ các đoạn cần khi xoay key ──────────────────────────────────
class TestRangeReupload:
    def test_signature_detection(self) -> None:
        def old_style(api_key: str) -> list:
            return []

        def new_style(api_key: str, ranges: list | None = None) -> list:
            return []

        def variadic(*args: Any) -> list:
            return []

        assert not _accepts_ranges_argument(old_style)
        assert _accepts_ranges_argument(new_style)
        assert _accepts_ranges_argument(variadic)

    def _translator(self, callback: Any) -> GeminiSubtitleTranslator:
        translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
        translator._video_handle_cache = {}
        # [v3.23.605] Ghi nhớ ref đã tải lại theo key dùng khoá trạng thái của adapter.
        translator._state_lock = threading.RLock()
        translator._video_handles_stale = True
        translator._api_key = "key-2"
        translator._video_reupload_cb = callback
        translator._resolve_video_handles = lambda refs: list(refs)
        return translator

    def test_heal_requests_only_needed_ranges(self) -> None:
        """Lời gọi chỉ đính đoạn 21 → callback chỉ được xin đúng khoảng của đoạn 21."""
        received: list[Any] = []

        def callback(api_key: str, ranges: list | None = None) -> list:
            received.append((api_key, ranges))
            return [SimpleNamespace(start_sec=7263.0, end_sec=7609.0, remote_name="new")]

        old_ref = SimpleNamespace(start_sec=7263.0, end_sec=7609.0, remote_name="old")
        refs = [old_ref]
        handles = self._translator(callback)._resolve_handles_with_heal(refs, [])

        assert received == [("key-2", [(7263.0, 7609.0)])]
        assert refs[0].remote_name == "new"
        assert handles and handles[0].remote_name == "new"

    def test_old_style_callback_still_works(self) -> None:
        calls: list[str] = []

        def callback(api_key: str) -> list:
            calls.append(api_key)
            return [SimpleNamespace(start_sec=0.0, end_sec=10.0, remote_name="new")]

        refs = [SimpleNamespace(start_sec=0.0, end_sec=10.0, remote_name="old")]
        self._translator(callback)._resolve_handles_with_heal(refs, [])
        assert calls == ["key-2"]

    def test_error_inside_new_callback_does_not_trigger_full_reupload(self) -> None:
        """TypeError BÊN TRONG callback mới không được hiểu nhầm thành "callback kiểu cũ"."""
        calls: list[int] = []

        def callback(api_key: str, ranges: list | None = None) -> list:
            calls.append(1)
            raise TypeError("lỗi nội bộ")

        refs = [SimpleNamespace(start_sec=0.0, end_sec=10.0, remote_name="old")]
        self._translator(callback)._resolve_handles_with_heal(refs, [])
        assert calls == [1]

    def test_provider_uploads_only_wanted_chunks(self, tmp_path: Path, monkeypatch) -> None:
        provider = GeminiVideoContextProvider(
            cache_db_path=tmp_path / "cache.db", work_dir=tmp_path
        )
        chunks = []
        for index in range(22):
            path = tmp_path / f"part{index:02d}.mp4"
            path.write_bytes(b"x")
            chunks.append(SimpleNamespace(
                index=index, path=path, start_sec=index * 346.0,
                end_sec=(index + 1) * 346.0, is_full_video=False,
            ))
        plan = SimpleNamespace(source_path=tmp_path / "movie.mp4", chunks=chunks,
                               needs_split=True)
        uploaded: list[int] = []
        monkeypatch.setattr(provider, "plan_chunks", lambda _path: plan)
        monkeypatch.setattr(
            "subtitles_extractor.infrastructure.translation.gemini_video_context.find_ffmpeg",
            lambda: "ffmpeg",
        )
        monkeypatch.setattr(provider, "_encode_missing_chunks", lambda *_a: [])
        monkeypatch.setattr(provider, "_chunk_source_key", lambda *_a: "k")

        def fake_upload(chunk: Any) -> Any:
            uploaded.append(chunk.index)
            return SimpleNamespace(remote_name=f"files/{chunk.index}")

        monkeypatch.setattr(provider, "upload_chunk", fake_upload)
        refs = provider.upload_chunks_for_ranges(
            tmp_path / "movie.mp4", [(21 * 346.0, 22 * 346.0)]
        )
        assert uploaded == [21]
        assert len(refs) == 1

    def test_provider_returns_empty_when_no_range_matches(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        provider = GeminiVideoContextProvider(
            cache_db_path=tmp_path / "cache.db", work_dir=tmp_path
        )
        plan = SimpleNamespace(source_path=tmp_path / "m.mp4", chunks=[
            SimpleNamespace(index=0, path=tmp_path / "a", start_sec=0.0, end_sec=10.0,
                            is_full_video=False),
        ], needs_split=False)
        monkeypatch.setattr(provider, "plan_chunks", lambda _path: plan)
        assert provider.upload_chunks_for_ranges(tmp_path / "m.mp4", [(500.0, 600.0)]) == []


# ── 2. Hạn sống cache upload ──────────────────────────────────────────────────
class TestUploadCacheLifetime:
    NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)

    def test_fresh_entry_is_valid(self) -> None:
        fresh = (self.NOW - timedelta(hours=2)).isoformat()
        assert not is_upload_expired(fresh, now=self.NOW)

    def test_entry_older_than_lifetime_is_expired(self) -> None:
        stale = (self.NOW - timedelta(hours=49)).isoformat()
        assert is_upload_expired(stale, now=self.NOW)

    def test_missing_or_bad_timestamp_keeps_old_behaviour(self) -> None:
        assert not is_upload_expired(None, now=self.NOW)
        assert not is_upload_expired("không phải ngày", now=self.NOW)

    def test_naive_timestamp_treated_as_utc(self) -> None:
        stale = (self.NOW - timedelta(hours=60)).replace(tzinfo=None).isoformat()
        assert is_upload_expired(stale, now=self.NOW)

    def test_lookup_drops_expired_row(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        provider = GeminiVideoContextProvider(cache_db_path=db, work_dir=tmp_path)
        provider._cache_store("k-old", "files/old")
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE video_context_uploads SET created_at=? WHERE file_hash='k-old'",
                ((datetime.now(UTC) - timedelta(days=5)).isoformat(),),
            )
        assert provider._cache_lookup("k-old") is None
        with sqlite3.connect(db) as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM video_context_uploads WHERE file_hash='k-old'"
            ).fetchone()[0]
        assert remaining == 0

    def test_lookup_keeps_fresh_row(self, tmp_path: Path) -> None:
        provider = GeminiVideoContextProvider(
            cache_db_path=tmp_path / "cache.db", work_dir=tmp_path
        )
        provider._cache_store("k-new", "files/new")
        assert provider._cache_lookup("k-new") == "files/new"


# ── 3. Đồng hồ canh không bắn nhầm ────────────────────────────────────────────
class _FakeTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _WatchdogHost:
    """Đủ thuộc tính để gọi ``_check_busy_watchdog`` mà không dựng cả ViewModel Qt."""

    def __init__(self) -> None:
        self._is_busy = True
        self._busy_watchdog = _FakeTimer()
        self._worker = SimpleNamespace(isRunning=lambda: False)
        self._analyze_worker = None
        self._busy_idle_ticks = 0
        self.unlocked = False

    def _set_busy(self, busy: bool) -> None:
        self._is_busy = busy
        self.unlocked = not busy


def _check(host: _WatchdogHost) -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    TranslatePageViewModel._check_busy_watchdog(host)  # type: ignore[arg-type]


class TestBusyWatchdog:
    def test_single_idle_tick_does_not_unlock(self) -> None:
        """Ca thật: tín hiệu kết thúc còn xếp hàng — nhịp đầu KHÔNG được mở khoá."""
        host = _WatchdogHost()
        _check(host)
        assert not host.unlocked

    def test_two_consecutive_idle_ticks_unlock(self) -> None:
        host = _WatchdogHost()
        _check(host)
        _check(host)
        assert host.unlocked

    def test_alive_worker_resets_counter(self) -> None:
        host = _WatchdogHost()
        _check(host)
        host._worker = SimpleNamespace(isRunning=lambda: True)
        _check(host)
        host._worker = SimpleNamespace(isRunning=lambda: False)
        _check(host)
        assert not host.unlocked


# ── 4. Thuật ngữ: phân tích phương án + phân loại ─────────────────────────────
class TestGlossaryParsing:
    def test_slash_inside_parentheses_is_not_split_first(self) -> None:
        assert glossary_alternatives("Thổ Cẩu (chó ta/chó cỏ)") == [
            "thổ cẩu", "chó ta", "chó cỏ",
        ]

    def test_plain_alternatives(self) -> None:
        assert glossary_alternatives("kẻ quê mùa / gã ngốc") == ["kẻ quê mùa", "gã ngốc"]

    def test_parenthetical_alternative_kept(self) -> None:
        assert "đại g" in glossary_alternatives("xe Mercedes G-Class (Đại G)")

    def test_real_case_no_longer_flagged(self) -> None:
        @dataclass
        class _E:
            index: int
            text: str
            start_sec: float = 0.0
            end_sec: float = 3.0

        source = [_E(1, "您以后叫我土狗就行")]
        translated = [_E(1, "Ngài cứ gọi em là Thổ Cẩu là được.")]
        flags = detect_quality_flags(
            source, translated, glossary="土狗 = Thổ Cẩu (chó ta/chó cỏ)"
        )
        assert flags["glossary_violations"] == []

    def test_kind_classification(self) -> None:
        assert glossary_entry_kind("Hồng Liên Nữ Đế") == GLOSSARY_KIND_NAME
        assert glossary_entry_kind("Tập đoàn Liễu Thị") == GLOSSARY_KIND_NAME
        assert glossary_entry_kind("Thức hải") == GLOSSARY_KIND_EXPRESSION
        assert glossary_entry_kind("lộng hành ngang ngược / tác oai tác quái") == (
            GLOSSARY_KIND_EXPRESSION
        )

    def test_headline_count_only_counts_names(self) -> None:
        @dataclass
        class _E:
            index: int
            text: str
            start_sec: float = 0.0
            end_sec: float = 3.0

        source = [_E(1, "他一向作威作福"), _E(2, "a"), _E(3, "柳氏集团的"), _E(4, "b")]
        translated = [
            _E(1, "Hắn vốn hống hách bá đạo."), _E(2, "x"),
            _E(3, "của nhà họ Liễu"), _E(4, "y"),
        ]
        glossary = "作威作福 = tác oai tác quái\n柳氏集团 = Tập đoàn Liễu Thị"
        counts = detect_quality_flags(source, translated, glossary=glossary)["counts"]
        assert counts["glossary_violations"] == 1
        assert counts["glossary_expression_deviations"] == 1


# ── 5. Bảng xưng hô mơ hồ ─────────────────────────────────────────────────────
class TestAmbiguousAddress:
    ROSTER = (
        "柳慕云 (Liễu Mộ Vân) [nữ, ~25] — nữ chính; xưng hô: với Tống Tiền tự xưng "
        "'tôi/Mộ Vân/chị' gọi 'cậu/Tống thần y/Tống Tiền', với Ông cụ Liễu xưng 'cháu' gọi "
        "'ông nội'\n"
        "宋钱 (Tống Tiền) [nam, ~22] — nhân vật chính; xưng hô: với Thiết Trụ xưng 'em' gọi "
        "'anh/Thiết Trụ ca', với Hứa Đào Hoa xưng 'em' gọi 'chị/chị dâu/Đào Hoa tỷ'\n"
        "向城 (Hướng Thành) [nam, ~45] — võ sư; xưng hô: với Tống Tiền trước khi thua xưng "
        "'ta/lão phu' gọi 'mày'\n"
        "许桃花 (Hứa Đào Hoa) [nữ] — góa phụ; xưng hô: với mẹ và em trai xưng 'tôi' gọi "
        "'mẹ/cậu'\n"
        "安雪漫 (An Tuyết Mạn) [nữ] — phản diện; xưng hô: với Tống Tiền xưng 'tao' gọi "
        "'mày/thằng ngốc/kẻ quê mùa'"
    )

    def test_real_case_detected(self) -> None:
        findings = ambiguous_address_rules(self.ROSTER)
        assert any("柳慕云" in item and "Tống Tiền" in item for item in findings)

    def test_one_pronoun_plus_title_is_fine(self) -> None:
        """'anh/Thiết Trụ ca' và 'chị/chị dâu/Đào Hoa tỷ' đều chỉ có MỘT đại từ."""
        findings = ambiguous_address_rules(self.ROSTER)
        assert not any("Thiết Trụ" in item for item in findings)
        assert not any("Hứa Đào Hoa" in item and "宋钱" in item for item in findings)

    def test_phase_conditioned_rule_is_fine(self) -> None:
        assert not any("向城" in item for item in ambiguous_address_rules(self.ROSTER))

    def test_group_addressee_is_fine(self) -> None:
        assert not any("mẹ và em trai" in item for item in ambiguous_address_rules(self.ROSTER))

    def test_insults_are_not_pronouns(self) -> None:
        assert not any("安雪漫" in item for item in ambiguous_address_rules(self.ROSTER))

    def test_analysis_prompt_demands_single_pronoun_pair(self) -> None:
        import inspect

        import subtitles_extractor.infrastructure.translation.gemini_translation_adapter as m

        source = inspect.getsource(m)
        assert "MỖI CẶP chỉ ghi ĐÚNG MỘT đại từ tự xưng" in source
        assert "KHÔNG " in source and "dùng '/' để liệt kê nhiều phương án" in source
