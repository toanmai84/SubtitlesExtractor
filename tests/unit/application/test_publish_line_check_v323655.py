"""[v3.23.655] Bug #12: trang Xuất bản cảnh báo phụ đề nhiều dòng hơn audio TTS."""

from __future__ import annotations

from pathlib import Path

from subtitles_extractor.application.services.publish_line_check import count_subtitle_lines, tts_line_mismatch


def _srt(path: Path, n: int) -> Path:
    path.write_text("".join(f"{i}\n00:00:0{i % 10},000 --> 00:00:0{i % 10},900\nDòng {i}\n\n" for i in range(1, n + 1)),
                    encoding="utf-8")
    return path


def test_mismatch_detected(tmp_path) -> None:
    sub = _srt(tmp_path / "ep.translate.vi.srt", 10)
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"x")
    _srt(tmp_path / "ep.tts.vi.srt", 7)
    found = tts_line_mismatch(sub, audio)
    assert found is not None and found.missing_audio == 3 and found.tts_lines == 7


def test_no_warning_when_synced_or_equal_or_unknown(tmp_path) -> None:
    audio = tmp_path / "ep.wav"
    audio.write_bytes(b"x")
    synced = _srt(tmp_path / "ep.tts.vi.srt", 7)
    assert tts_line_mismatch(synced, audio) is None                       # đang dùng chính phụ đề đồng bộ
    assert tts_line_mismatch(_srt(tmp_path / "a.srt", 7), audio) is None   # cùng số dòng
    assert tts_line_mismatch(_srt(tmp_path / "b.srt", 9), tmp_path / "other.wav") is None  # audio không có phụ đề kèm


def test_count_ass(tmp_path) -> None:
    ass = tmp_path / "x.ass"
    ass.write_text("[Events]\nFormat: Layer\nDialogue: 0,a\nDialogue: 0,b\n", encoding="utf-8")
    assert count_subtitle_lines(ass) == 2


def test_publish_page_warns_and_switches_to_synced(tmp_path, monkeypatch) -> None:
    import pytest

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QMessageBox

    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import JsonFileRepository
    from subtitles_extractor.presentation.pages import publish_page as module

    QApplication.instance() or QApplication([])
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "c.json"),
        i18n_data_dir=Path(module.__file__).resolve().parents[2] / "data", user_data_dir=tmp_path,
    )
    page = module.PublishPage(container)
    try:
        sub = _srt(tmp_path / "ep.translate.vi.srt", 10)
        audio = tmp_path / "ep.wav"
        audio.write_bytes(b"x")
        synced = _srt(tmp_path / "ep.tts.vi.srt", 7)
        voice_over = next(i for i, (_l, mode) in enumerate(module._AUDIO_CHOICES) if mode is module.AudioMode.VOICE_OVER)
        page._audio_combo.setCurrentIndex(voice_over)
        page._subtitle_edit.setText(str(sub))
        page._audio_edit.setText(str(audio))
        page._update_readiness()
        assert "≠" in page._sync_note()
        asked: list = []
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.Cancel)
        page._start_render()
        assert asked and page._thread is None                      # Huỷ → không xuất
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        assert page._confirm_line_mismatch()
        assert page._subtitle_edit.text() == str(synced)           # đã chuyển sang phụ đề đồng bộ
        assert page._line_mismatch() is None
    finally:
        page.close()
