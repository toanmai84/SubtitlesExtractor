"""Test chọn nguồn phụ đề khi xuất bản — v3.23.318.

LỖI ĐƯỢC SỬA: khâu TTS chỉnh lại mốc thời gian phụ đề cho khớp lời thoại đã tổng hợp
và ghi ra ``<tên>.tts.<lang>.srt``. Nhưng trang Xuất bản lại lấy phụ đề từ trang Dịch
(mốc GỐC) → phim xuất ra có phụ đề LỆCH so với giọng thuyết minh, càng về sau càng lệch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.publish_subtitle_selector import (
    SubtitleSource,
    choose_publish_subtitle,
    find_tts_synced_subtitle,
)
from subtitles_extractor.domain.value_objects.output_naming import tts_subtitle_path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Trả về ``(tệp âm thanh TTS, tệp phụ đề bản dịch)``."""
    audio = tmp_path / "phim.wav"
    audio.write_bytes(b"\x00")
    translated = tmp_path / "ban_dich.srt"
    translated.write_text("1\n", encoding="utf-8")
    return audio, translated


# ── Tìm phụ đề đồng bộ TTS ───────────────────────────────────────────────────
def test_finds_language_specific_variant(workspace: tuple[Path, Path]) -> None:
    audio, _ = workspace
    expected = tts_subtitle_path(audio, "vi")
    expected.write_text("1\n", encoding="utf-8")
    assert find_tts_synced_subtitle(audio, "vi") == expected


def test_falls_back_to_variant_without_language(workspace: tuple[Path, Path]) -> None:
    """Tệp được đặt tên kèm mã ngôn ngữ chỉ khi biết ngôn ngữ lúc tổng hợp."""
    audio, _ = workspace
    plain = tts_subtitle_path(audio, "")
    plain.write_text("1\n", encoding="utf-8")
    assert find_tts_synced_subtitle(audio, "vi") == plain


def test_returns_none_when_no_synced_file(workspace: tuple[Path, Path]) -> None:
    audio, _ = workspace
    assert find_tts_synced_subtitle(audio, "vi") is None


def test_returns_none_without_audio() -> None:
    assert find_tts_synced_subtitle(None, "vi") is None


# ── Chọn nguồn ───────────────────────────────────────────────────────────────
def test_prefers_tts_synced_over_translated(workspace: tuple[Path, Path]) -> None:
    """ĐÂY LÀ LỖI ĐƯỢC SỬA: phải ưu tiên bản đã chỉnh giờ, không dùng bản dịch."""
    audio, translated = workspace
    synced = tts_subtitle_path(audio, "vi")
    synced.write_text("1\n", encoding="utf-8")

    choice = choose_publish_subtitle(
        tts_audio_path=audio,
        translated_subtitle_path=translated,
        target_language="vi",
        audio_will_be_used=True,
    )

    assert choice.source is SubtitleSource.TTS_SYNCED
    assert choice.path == synced
    assert choice.warning is None


def test_uses_translated_when_tts_not_run(workspace: tuple[Path, Path]) -> None:
    """Chưa chạy TTS thì bản dịch là đúng — KHÔNG được cảnh báo thừa."""
    _, translated = workspace
    choice = choose_publish_subtitle(
        tts_audio_path=None, translated_subtitle_path=translated
    )
    assert choice.source is SubtitleSource.TRANSLATED
    assert choice.path == translated
    assert choice.warning is None


def test_warns_when_audio_used_but_synced_missing(
    workspace: tuple[Path, Path],
) -> None:
    """Có giọng đọc mà thiếu phụ đề đồng bộ -> phải CẢNH BÁO nguy cơ lệch."""
    audio, translated = workspace
    choice = choose_publish_subtitle(
        tts_audio_path=audio,
        translated_subtitle_path=translated,
        target_language="vi",
        audio_will_be_used=True,
    )
    assert choice.source is SubtitleSource.TRANSLATED
    assert choice.warning is not None
    assert "LỆCH" in choice.warning


def test_no_warning_when_audio_not_used(workspace: tuple[Path, Path]) -> None:
    """Xuất chỉ có phụ đề (không kèm giọng) thì mốc gốc là đúng — không cảnh báo."""
    audio, translated = workspace
    choice = choose_publish_subtitle(
        tts_audio_path=audio,
        translated_subtitle_path=translated,
        target_language="vi",
        audio_will_be_used=False,
    )
    assert choice.warning is None


def test_none_when_nothing_available() -> None:
    choice = choose_publish_subtitle(
        tts_audio_path=None, translated_subtitle_path=None
    )
    assert choice.source is SubtitleSource.NONE
    assert choice.path is None


def test_naming_convention_matches_generate_tts(workspace: tuple[Path, Path]) -> None:
    """Quy ước tên phải khớp thứ ``generate_tts`` thực sự ghi ra."""
    audio, _ = workspace
    assert tts_subtitle_path(audio, "vi").name == "phim.tts.vi.srt"
    assert tts_subtitle_path(audio, "").name == "phim.tts.srt"
    # Mã ngôn ngữ dạng "vi-VN" phải rút gọn về "vi".
    assert tts_subtitle_path(audio, "vi-VN").name == "phim.tts.vi.srt"


# ── [v3.23.323] Hồi quy từ LOG THẬT trên máy người dùng ──────────────────────
def test_finds_language_variant_without_knowing_language(tmp_path: Path) -> None:
    """LỖI THẬT: TTS tạo ``第19集.tts.vi.srt`` nhưng chọn nguồn báo "không tìm thấy".

    Nguyên nhân: hàm ĐOÁN mã ngôn ngữ từ ``ProjectRecord.target_lang``; khi trường này
    rỗng (dự án chưa qua khâu Dịch trong phiên đó) thì chỉ thử ``.tts.srt`` và bỏ sót
    tệp thật. Nay quét mẫu ``*.tts.*.srt`` nên không phụ thuộc việc biết trước.
    """
    audio = tmp_path / "第19集.flac"
    audio.write_bytes(b"\x00")
    synced = tmp_path / "第19集.tts.vi.srt"
    synced.write_text("1\n", encoding="utf-8")

    # target_language RỖNG — đúng tình huống trong log.
    assert find_tts_synced_subtitle(audio, "") == synced


def test_chooses_synced_without_warning_when_language_unknown(tmp_path: Path) -> None:
    """Hệ quả của lỗi trên: người dùng bị cảnh báo LỆCH dù tệp đúng vẫn tồn tại."""
    audio = tmp_path / "第19集.flac"
    audio.write_bytes(b"\x00")
    (tmp_path / "第19集.tts.vi.srt").write_text("1\n", encoding="utf-8")
    translated = tmp_path / "ban_dich.srt"
    translated.write_text("1\n", encoding="utf-8")

    choice = choose_publish_subtitle(
        tts_audio_path=audio,
        translated_subtitle_path=translated,
        target_language="",          # không biết ngôn ngữ
        audio_will_be_used=True,
    )

    assert choice.source is SubtitleSource.TTS_SYNCED
    assert choice.warning is None    # KHÔNG được cảnh báo sai


def test_prefers_requested_language_when_several_variants(tmp_path: Path) -> None:
    audio = tmp_path / "phim.flac"
    audio.write_bytes(b"\x00")
    (tmp_path / "phim.tts.en.srt").write_text("1\n", encoding="utf-8")
    vietnamese = tmp_path / "phim.tts.vi.srt"
    vietnamese.write_text("1\n", encoding="utf-8")

    assert find_tts_synced_subtitle(audio, "vi") == vietnamese


def test_scan_is_deterministic(tmp_path: Path) -> None:
    """Nhiều biến thể mà không biết ngôn ngữ -> kết quả phải ỔN ĐỊNH giữa các lần gọi."""
    audio = tmp_path / "phim.flac"
    audio.write_bytes(b"\x00")
    for lang in ("vi", "en", "ja"):
        (tmp_path / f"phim.tts.{lang}.srt").write_text("1\n", encoding="utf-8")

    first = find_tts_synced_subtitle(audio, "")
    assert first is not None
    for _ in range(5):
        assert find_tts_synced_subtitle(audio, "") == first
