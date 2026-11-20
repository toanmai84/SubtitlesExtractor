"""[v3.23.738] Rà soát cài đặt: mọi cài đặt phải có tác dụng; hằng số cần chỉnh đưa ra Cài đặt (chủ dự án, 17/11/2026).

Đo bằng cách đổi từng cài đặt và xem thứ ứng dụng thật sự dùng (yêu cầu trích xuất, cấu hình bộ dựng / OCR / MPV / dò
ROI, bộ dịch) + phát lại phim thật với từng tham số bộ dựng. Kết quả:

* "Định dạng đầu ra" (``post_process.output_format``) không chỗ nào đọc → nay là định dạng mặc định khi Xuất tệp ở trang
  Biên tập (tệp nội bộ ``.original.srt`` của các khâu hàng loạt giữ SRT — chuỗi cả bộ nhận ra tập đã trích qua tệp đó);
* tắt "Hiện dạng sóng" lúc đang chạy để lại thanh phóng dọc dạng sóng;
* đổi ngôn ngữ giao diện không báo cần khởi động lại;
* soát câu nghi ngờ: > 200 câu thì BỎ soát cả lượt (hằng số) → nay soát câu đáng ngờ nhất trước tới giới hạn (Cài đặt);
* ngưỡng tin cậy để PaddleOCR-VL soát lại (0,90) và chất lượng video ROI (24) đưa ra Cài đặt.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

DATA = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor" / "data"


def test_new_settings_have_defaults_and_bounds() -> None:
    from pydantic import ValidationError

    from subtitles_extractor.infrastructure.settings.application_settings import (
        ApplicationSettings,
    )

    s = ApplicationSettings()
    assert s.ocr.verify_max_lines == 400 and s.ocr.vl_low_confidence == pytest.approx(0.90)
    assert s.advanced.roi_clip_quality == 24
    with pytest.raises(ValidationError):
        ApplicationSettings.model_validate({"ocr": {"vl_low_confidence": 1.5}})


def test_verify_cap_keeps_most_suspect_lines_instead_of_skipping_all() -> None:
    from subtitles_extractor.application.use_cases.extract_subtitles import (
        prioritize_suspects,
    )

    events = [SimpleNamespace(confidence=c) for c in (0.99, 0.5, 0.95, 0.7, 0.88)]
    suspects = [(0, "missing_char"), (1, "low_confidence"), (2, "reads_disagree"), (3, "low_confidence"),
                (4, "low_confidence")]
    kept = prioritize_suspects(suspects, events, 3)
    # Cờ hình học / hai bản đọc trước, rồi câu tin cậy THẤP NHẤT; giữ thứ tự câu khi trả về.
    assert kept == [(0, "missing_char"), (1, "low_confidence"), (2, "reads_disagree")]
    assert prioritize_suspects(suspects, events, 10) == suspects


def test_request_carries_new_settings() -> None:
    from subtitles_extractor.application.dtos.extract_subtitles_dto import (
        ExtractSubtitlesRequest,
    )

    fields = ExtractSubtitlesRequest.__dataclass_fields__
    assert {"verify_max_lines", "vl_low_confidence", "roi_clip_quality"} <= set(fields)


def test_roi_clip_quality_reaches_encoder_args() -> None:
    import inspect

    from subtitles_extractor.infrastructure.video.roi_clip_exporter import (
        FfmpegRoiClipExporter,
    )

    assert "quality" in inspect.signature(FfmpegRoiClipExporter.export).parameters


def test_export_dialog_follows_output_format_setting() -> None:
    from subtitles_extractor.domain.value_objects.device_kind import SubtitleFormat
    from subtitles_extractor.presentation.pages.editor_export import (
        export_dialog_defaults,
    )

    path, selected = export_dialog_defaults(Path("/v/phim.mp4"), SubtitleFormat.ASS,
                                            "SubRip (*.srt);;Advanced SubStation Alpha (*.ass)")
    assert path == Path("/v/phim.original.ass") and selected == "Advanced SubStation Alpha (*.ass)"
    path, selected = export_dialog_defaults(Path("/v/phim.mp4"), SubtitleFormat.SRT,
                                            "SubRip (*.srt);;Advanced SubStation Alpha (*.ass)")
    assert path == Path("/v/phim.original.srt") and selected == "SubRip (*.srt)"


def test_short_text_label_says_latin_only() -> None:
    for lang, word in (("vi", "Latin"), ("en", "Latin")):
        strings = json.loads((DATA / f"strings_{lang}.json").read_text(encoding="utf-8"))
        assert word in strings["settings"]["rl_drop_short"]
        assert "msg_restart_locale" in strings["settings"]


def test_new_settings_on_settings_page_saved_and_loaded() -> None:
    page = (Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages/settings_page.py"
            ).read_text(encoding="utf-8")
    for key in ("verify_max_lines", "vl_low_confidence", "roi_clip_quality"):
        assert f'"{key}"' in page, key                       # được lưu
        assert f".{key}" in page, key                         # được nạp


def test_waveform_toggle_hides_zoom_too() -> None:
    page = (Path(__file__).resolve().parents[3] / "src/subtitles_extractor/presentation/pages/editor_page.py"
            ).read_text(encoding="utf-8")
    body = page.split("def apply_ui_settings", 1)[1].split("\n    def ", 1)[0]
    assert "_waveform_y_zoom.setVisible(show_waveform)" in body


def test_every_setting_is_read_outside_settings_page() -> None:
    """Chặn cài đặt "để cho có" mới: mỗi trường phải được ĐỌC ở ít nhất một nơi ngoài định nghĩa và trang Cài đặt."""
    import re

    from subtitles_extractor.infrastructure.settings.application_settings import (
        ApplicationSettings,
    )

    root = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"
    sources = [p.read_text(encoding="utf-8") for p in root.rglob("*.py")
               if p.name not in ("application_settings.py", "settings_page.py")]
    dead = []
    for section, info in ApplicationSettings.model_fields.items():
        model = info.annotation
        for name in getattr(model, "model_fields", {}):
            # Một tệp "đọc" trường khi nó truy cập NHÓM cài đặt đó (``snapshot.ocr``, ``.translation`` gán biến) và nhắc
            # tên trường (``.tên`` / ``"tên"``). Trường chỉ trùng tên với trường DTO khác (``request.output_format``)
            # mà không tệp nào chạm nhóm của nó thì bị bắt — đúng lỗi "Định dạng đầu ra" trước 3.23.738.
            name_pat = re.compile(rf"\.{name}\b|[\"']{name}[\"']")
            section_pat = re.compile(rf"\.{section}\b|[\"']{section}[\"']")
            if not any(section_pat.search(src) and name_pat.search(src) for src in sources):
                dead.append(f"{section}.{name}")
    assert dead == []
