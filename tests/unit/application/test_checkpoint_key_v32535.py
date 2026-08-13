"""[v3.23.135] Test: khóa checkpoint phản ánh MỌI cài đặt ảnh hưởng kết quả dịch.

Đổi phong cách / locale / thinking / context_size / visual-cues => khóa PHẢI đổi (để không
resume nhầm bản dịch cũ). Cùng cấu hình => khóa GIỮ NGUYÊN (resume hoạt động).
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    _compute_checkpoint_key,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval


def _events() -> list[SubtitleEvent]:
    return [SubtitleEvent(index=1, text="你好", interval=TimeInterval(0.0, 1.0))]


def _stage(**over) -> TranslationStageConfig:
    base = {
        "kind": TranslationStageKind.STYLE,
        "model_name": "m",
        "batch_size": 50,
        "temperature": 0.3,
    }
    base.update(over)
    return TranslationStageConfig(**base)


def _req(stage: TranslationStageConfig, **over) -> TranslateSubtitlesRequest:
    base = {
        "events": _events(),
        "stages": [stage],
        "context": TranslationContext(target_lang="Vietnamese", source_lang="zh"),
    }
    base.update(over)
    return TranslateSubtitlesRequest(**base)


def test_same_config_same_key() -> None:
    k1 = _compute_checkpoint_key(_req(_stage()))
    k2 = _compute_checkpoint_key(_req(_stage()))
    assert k1 == k2


def test_style_name_changes_key() -> None:
    k1 = _compute_checkpoint_key(_req(_stage(style_name="Trang trọng")))
    k2 = _compute_checkpoint_key(_req(_stage(style_name="Thân mật")))
    assert k1 != k2


def test_locale_notes_changes_key() -> None:
    k1 = _compute_checkpoint_key(_req(_stage(locale_notes="Miền Bắc")))
    k2 = _compute_checkpoint_key(_req(_stage(locale_notes="Miền Nam")))
    assert k1 != k2


def test_thinking_level_changes_key() -> None:
    k1 = _compute_checkpoint_key(_req(_stage(thinking_level="low")))
    k2 = _compute_checkpoint_key(_req(_stage(thinking_level="high")))
    assert k1 != k2


def test_context_size_changes_key() -> None:
    k1 = _compute_checkpoint_key(_req(_stage(context_size=5)))
    k2 = _compute_checkpoint_key(_req(_stage(context_size=20)))
    assert k1 != k2


def test_visual_cues_changes_key() -> None:
    base = _stage()
    k_off = _compute_checkpoint_key(_req(base, enable_visual_cues=False))
    k_on = _compute_checkpoint_key(_req(base, enable_visual_cues=True))
    assert k_off != k_on


def test_visual_cues_batch_size_changes_key() -> None:
    base = _stage()
    k1 = _compute_checkpoint_key(
        _req(base, enable_visual_cues=True, visual_cues_batch_size=100)
    )
    k2 = _compute_checkpoint_key(
        _req(base, enable_visual_cues=True, visual_cues_batch_size=200)
    )
    assert k1 != k2


def test_source_lang_still_distinguishes() -> None:
    # Hồi quy: vẫn phân biệt ngôn ngữ gốc như trước.
    r1 = _req(_stage())
    r2 = _req(
        _stage(),
        context=TranslationContext(target_lang="Vietnamese", source_lang="ko"),
    )
    assert _compute_checkpoint_key(r1) != _compute_checkpoint_key(r2)


def test_temperature_still_changes_key() -> None:
    # Hồi quy: tham số cũ vẫn còn hiệu lực.
    assert _compute_checkpoint_key(
        _req(_stage(temperature=0.1))
    ) != _compute_checkpoint_key(_req(_stage(temperature=0.9)))
