"""[v3.23.135] Test: checkpoint phản ánh MỌI cài đặt ảnh hưởng kết quả dịch.

Đổi phong cách / locale / thinking / context_size / visual-cues => KHÔNG được resume nhầm
bản dịch cũ. Cùng cấu hình => resume hoạt động.

[v3.23.606] Cấu hình TỪNG GIAI ĐOẠN chuyển từ khoá file sang dấu vân tay riêng của giai
đoạn (:func:`stage_config_signature`) — đổi model một giai đoạn không còn làm mất mọi
giai đoạn đã xong. Khoá file chỉ còn phụ đề, cặp ngôn ngữ và Visual Cues.
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    _compute_checkpoint_key,
    stage_config_signature,
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
    k1 = stage_config_signature(_stage(style_name="Trang trọng"), attach_video=False)
    k2 = stage_config_signature(_stage(style_name="Thân mật"), attach_video=False)
    assert k1 != k2


def test_locale_notes_changes_key() -> None:
    k1 = stage_config_signature(_stage(locale_notes="Miền Bắc"), attach_video=False)
    k2 = stage_config_signature(_stage(locale_notes="Miền Nam"), attach_video=False)
    assert k1 != k2


def test_thinking_level_changes_key() -> None:
    """[v3.23.609] Mức suy nghĩ chỉ được gửi tới model khi BẬT suy nghĩ — xét khi bật."""
    k1 = stage_config_signature(
        _stage(enable_thinking=True, thinking_level="low"), attach_video=False
    )
    k2 = stage_config_signature(
        _stage(enable_thinking=True, thinking_level="high"), attach_video=False
    )
    assert k1 != k2


def test_thinking_level_ignored_while_thinking_is_off() -> None:
    """[v3.23.609] Suy nghĩ tắt: mức suy nghĩ không được gửi → không đổi dấu vân tay."""
    k1 = stage_config_signature(_stage(thinking_level="low"), attach_video=False)
    k2 = stage_config_signature(_stage(thinking_level="high"), attach_video=False)
    assert k1 == k2


def test_context_size_changes_key() -> None:
    k1 = stage_config_signature(_stage(context_size=5), attach_video=False)
    k2 = stage_config_signature(_stage(context_size=20), attach_video=False)
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
    assert stage_config_signature(
        _stage(temperature=0.1), attach_video=False
    ) != stage_config_signature(_stage(temperature=0.9), attach_video=False)


def test_attach_video_changes_stage_signature() -> None:
    stage = _stage()
    assert stage_config_signature(stage, attach_video=True) != stage_config_signature(
        stage, attach_video=False
    )


def test_model_change_keeps_file_key_and_stage_signature() -> None:
    """[v3.23.606] Model được so theo hạng khi dùng lại, không nằm trong khoá/vân tay."""
    weak = _stage(model_name="gemini-3.5-flash-lite")
    strong = _stage(model_name="gemini-3.8-flash")
    assert _compute_checkpoint_key(_req(weak)) == _compute_checkpoint_key(_req(strong))
    assert stage_config_signature(weak, attach_video=False) == stage_config_signature(
        strong, attach_video=False
    )
