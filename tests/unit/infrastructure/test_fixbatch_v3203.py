"""Test [v3.20.3] gói 16 fix — phần logic thuần kiểm thử được headless."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSrtTolerantParser:
    """#5 — Trình đọc SRT bao dung: mũi tên hỏng + ký tự tàng hình."""

    @staticmethod
    def _parse(raw: str):
        from subtitles_extractor.infrastructure.subtitle.importers.srt_importer import (
            _parse_srt,
        )
        return list(_parse_srt(raw))

    @pytest.mark.parametrize(
        "raw",
        [
            "1\n00:00:01,000 -> 00:00:02,000\nA\n",          # arrow ->
            "1\n00:00:01,000 -- > 00:00:02,000\nB\n",        # arrow -- >
            "1\n00:00:01,000 \u2014> 00:00:02,000\nC\n",     # em dash —>
            "1\n00:00:01,000 \u2013> 00:00:02,000\nD\n",     # en dash –>
            "1\n00:00:01,000\u200b --> 00:00:02,000\nE\u200b\n",  # zero-width
            "1\n00:00:01,000\u00a0-->\u00a000:00:02,000\nF\n",    # nbsp
        ],
    )
    def test_garbage_srt_still_parses_one_cue(self, raw: str) -> None:
        events = self._parse(raw)
        assert len(events) == 1
        assert abs(events[0].start_sec - 1.0) < 0.01
        assert abs(events[0].end_sec - 2.0) < 0.01


class TestJsonHallucinationShield:
    """#3 — Bóc JSON khỏi văn bản 'lắm lời' của AI."""

    @staticmethod
    def _clean(raw: str) -> str:
        from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
            _sanitize_json_text,
        )
        return _sanitize_json_text(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            '{"a":1}',
            '```json\n{"a":1}\n```',
            'Dạ, kết quả đây: {"a":1}',
            'json {"a":1}',
            'Đây ạ: [1,2,3]',
            '{"a":1} . Hy vọng giúp được bạn!',
            'prefix {"a":{"b":[1,2]}} suffix',
        ],
    )
    def test_extracted_text_is_valid_json(self, raw: str) -> None:
        json.loads(self._clean(raw))  # không ném JSONDecodeError là đạt


class TestEdgeTtsJitterImport:
    """#3net — module edge_tts có random để jitter backoff."""

    def test_random_imported(self) -> None:
        import subtitles_extractor.infrastructure.tts.edge_tts_adapter as mod

        assert hasattr(mod, "random")


class TestAtomicSaveGcCollect:
    """#2 — atomic_save gọi gc.collect trước os.replace (lách Defender)."""

    def test_atomic_write_still_works(self, tmp_path: Path) -> None:
        from subtitles_extractor.infrastructure.subtitle.atomic_save import (
            atomic_write_text,
        )

        target = tmp_path / "out.srt"
        atomic_write_text(target, "nội dung\nhai dòng")
        assert target.read_text(encoding="utf-8") == "nội dung\nhai dòng"

    def test_gc_imported(self) -> None:
        import subtitles_extractor.infrastructure.subtitle.atomic_save as mod

        assert hasattr(mod, "gc")


class TestSqliteCloseUsesPassive:
    """#2 — close() dùng wal_checkpoint(PASSIVE) thay TRUNCATE."""

    def test_source_uses_passive_on_close(self) -> None:
        import inspect

        from subtitles_extractor.infrastructure.database.sqlite_subtitle_repository import (
            SqliteSubtitleRepository,
        )

        src = inspect.getsource(SqliteSubtitleRepository.close)
        # Câu lệnh PRAGMA thực thi phải là PASSIVE (docstring có thể nhắc TRUNCATE
        # để giải thích lý do đổi — nên chỉ kiểm dòng execute).
        exec_lines = [ln for ln in src.splitlines() if "wal_checkpoint" in ln and "execute" in ln]
        assert exec_lines, "Không tìm thấy lệnh execute wal_checkpoint trong close()"
        assert all("PASSIVE" in ln for ln in exec_lines)
        assert all("TRUNCATE" not in ln for ln in exec_lines)
