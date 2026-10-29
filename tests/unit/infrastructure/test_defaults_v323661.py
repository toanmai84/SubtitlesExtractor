"""[v3.23.661] Mặc định mới đo thật trên phim mẫu + nâng cấu hình cũ đúng một lần (giữ lựa chọn riêng của người dùng)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings
from subtitles_extractor.infrastructure.settings.settings_service import SettingsService
from subtitles_extractor.infrastructure.video import video_render_command as vrc


class _Repo:
    def __init__(self, data):
        self.data = data

    def load(self, key, default=None):
        return self.data if self.data is not None else default

    def save(self, key, value):
        self.data = value


def _old_saved(**frame_ocr):
    base = ApplicationSettings().model_dump(mode="python")
    base["frame"].pop("defaults_v661", None)
    base["ocr"].pop("defaults_v661", None)
    base["frame"]["sample_step_sec"] = 0.04
    base["ocr"].update(version="PP-OCRv6_medium", detection_model="PP-OCRv6_medium_det",
                       recognition_model="PP-OCRv6_medium_rec")
    for group, values in frame_ocr.items():
        base[group].update(values)
    return base


def test_fresh_defaults() -> None:
    s = ApplicationSettings()
    assert s.frame.sample_step_sec == 0.12
    assert (s.ocr.version, s.ocr.detection_model, s.ocr.recognition_model) == (
        "PP-OCRv6_small", "PP-OCRv6_small_det", "PP-OCRv6_small_rec")
    assert vrc._DEFAULT_CRF == 23


def test_old_defaults_are_upgraded_once() -> None:
    svc = SettingsService(_Repo(_old_saved()))
    cur = svc.current
    assert cur.frame.sample_step_sec == 0.12 and cur.ocr.version == "PP-OCRv6_small"
    assert cur.ocr.recognition_model == "PP-OCRv6_small_rec"


def test_user_choice_is_kept() -> None:
    svc = SettingsService(_Repo(_old_saved(frame={"sample_step_sec": 0.08}, ocr={"version": "PP-OCRv5_server",
                                                                              "detection_model": "PP-OCRv5_server_det",
                                                                              "recognition_model": "PP-OCRv5_server_rec"})))
    cur = svc.current
    assert cur.frame.sample_step_sec == 0.08 and cur.ocr.version == "PP-OCRv5_server"


def test_choice_after_upgrade_is_not_overridden_again() -> None:
    data = _old_saved()
    data["frame"]["defaults_v661"] = True
    data["ocr"]["defaults_v661"] = True  # đã nâng ở lần mở trước; người dùng tự chọn lại medium @0,04
    svc = SettingsService(_Repo(data))
    assert svc.current.frame.sample_step_sec == 0.04 and svc.current.ocr.version == "PP-OCRv6_medium"
