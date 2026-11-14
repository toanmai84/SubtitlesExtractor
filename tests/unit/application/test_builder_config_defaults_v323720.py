"""[v3.23.720] Cấu hình dựng phụ đề: một nguồn duy nhất, mặc định đệm đuôi câu = 0.

Lỗi gốc: 4 bản chép tay ánh xạ cài đặt → ``SubtitleBuilderConfig`` (container, Trích xuất, Chỉnh sửa/ReOCR, Debug), còn
các công cụ phát lại / thử sức / hiệu chỉnh dùng mặc định DTO — lệch mặc định ứng dụng ở 5 trường (khoảng gộp 0,6 / 0,2,
viterbi tắt / bật, số chữ tối thiểu 2 / 0, đệm 0,05 / 0, bước lấy mẫu) → số đo phát lại là của cấu hình không ai chạy.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from subtitles_extractor.application.dtos.extract_subtitles_dto import SubtitleBuilderConfig
from subtitles_extractor.composition.builder_config import app_default_builder_config, builder_config_from_settings
from subtitles_extractor.infrastructure.settings.application_settings import ApplicationSettings

_SRC = Path(__file__).resolve().parents[3] / "src" / "subtitles_extractor"
_TOOLS = Path(__file__).resolve().parents[3] / "tools"


def test_dto_padding_default_is_zero_like_the_app() -> None:
    assert SubtitleBuilderConfig().temporal_padding_sec == 0.0
    assert ApplicationSettings().post_process.temporal_padding_sec == 0.0


def test_app_default_config_follows_the_settings_defaults() -> None:
    settings = ApplicationSettings()
    config = app_default_builder_config()
    assert config.merge_gap_sec == settings.post_process.merge_gap_sec
    assert config.use_viterbi == settings.post_process.use_viterbi
    assert config.min_text_chars == settings.threshold.drop_short_text_chars
    assert config.temporal_padding_sec == settings.post_process.temporal_padding_sec
    assert config.sample_step_sec == settings.frame.sample_step_sec
    assert app_default_builder_config(sample_step_sec=0.04).sample_step_sec == 0.04


def test_every_builder_field_comes_from_the_settings() -> None:
    """Mọi trường của DTO đều được ánh xạ (trường mới thêm mà quên ánh xạ thì rơi về mặc định DTO, lệch ứng dụng)."""
    settings = ApplicationSettings()
    settings.post_process.temporal_padding_sec = 0.37
    settings.post_process.merge_gap_sec = 0.41
    settings.post_process.alignment_center_tolerance_ratio = 0.07
    config = builder_config_from_settings(settings, min_confidence=0.61)
    assert (config.temporal_padding_sec, config.merge_gap_sec, config.alignment_center_tolerance_ratio) == (0.37, 0.41, 0.07)
    assert config.min_confidence == 0.61
    mapped = set(ast.dump(ast.parse((_SRC / "composition" / "builder_config.py").read_text(encoding="utf-8"))).split("'"))
    missing = [field.name for field in dataclasses.fields(SubtitleBuilderConfig) if field.name not in mapped]
    assert missing == []


def test_no_hand_written_mapping_left_in_the_app() -> None:
    """Không còn chỗ nào trong ứng dụng tự dựng ``SubtitleBuilderConfig(...)`` từ cài đặt ngoài nguồn chung."""
    offenders = []
    for path in _SRC.rglob("*.py"):
        if path.name == "builder_config.py" or "application" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "SubtitleBuilderConfig":
                offenders.append(f"{path.relative_to(_SRC)}:{node.lineno}")
    assert offenders == []


def test_tools_build_with_app_defaults_when_the_seraw_has_no_run_record() -> None:
    for name in ("stress_subtitle_builder.py", "replay_subtitle_builder.py", "calibration/cli.py"):
        assert "app_default_builder_config" in (_TOOLS / name).read_text(encoding="utf-8"), name
