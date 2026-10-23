"""[v3.23.502] MỌI giai đoạn dịch đều phải có quy tắc căn dòng 1:1 + bước tự kiểm.

Đo trên bản dịch thật: 62 dòng bị LỆCH nội dung, có ca "睁大你们的狗眼" (mở to mắt ra mà nhìn) bị
dịch thành "Maybach" — nội dung của dòng khác lọt sang.

Trước v3.23.502, giai đoạn PREPROCESS THIẾU HẲN các quy tắc này dù nó chuẩn hoá dòng NGUỒN — lệch ở
khâu đó kéo lệch toàn bộ các giai đoạn sau. Test duyệt TỪNG giá trị của enum nên giai đoạn thêm mới
sau này cũng bị bắt buộc tuân thủ.
"""
from __future__ import annotations

import pytest

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

REQUIRED_RULES = [
    "CĂN DÒNG 1:1",
    "CẤM MƯỢN NỘI DUNG DÒNG KHÁC",
    "KHÔNG BỎ NGẮN BẤT THƯỜNG",
    "TỰ KIỂM TRƯỚC KHI TRẢ KẾT QUẢ",
]


def _instruction(kind: TranslationStageKind) -> str:
    translator = GeminiSubtitleTranslator.__new__(GeminiSubtitleTranslator)
    for attribute, value in (
        ("_analysis_media_resolution", "medium"),
        ("_emotion_cues", False),
        ("_reading_cps_budget", 0.0),
        ("_active_glossary", ""),
    ):
        setattr(translator, attribute, value)
    context = TranslationContext(
        target_lang="Vietnamese",
        characters="林逸 (Lâm Dật) [nam] — nhân vật chính",
        glossary="奔驰 = Mercedes-Benz",
        overview="tóm tắt",
    )
    stage = TranslationStageConfig(kind=kind, model_name="gemini-3.5-flash")
    return translator._system_instruction(stage, context, has_prior=False)


@pytest.mark.parametrize("kind", list(TranslationStageKind), ids=lambda k: k.name)
@pytest.mark.parametrize("rule", REQUIRED_RULES)
def test_every_stage_has_alignment_rules(kind: TranslationStageKind, rule: str) -> None:
    assert rule in _instruction(kind), f"giai đoạn {kind.name} thiếu quy tắc '{rule}'"


@pytest.mark.parametrize("kind", list(TranslationStageKind), ids=lambda k: k.name)
def test_every_stage_forbids_reordering(kind: TranslationStageKind) -> None:
    text = _instruction(kind)
    assert "Giữ nguyên tuyệt đối line_no" in text
    assert "không đổi thứ tự" in text


def test_stage_enum_not_empty() -> None:
    """Chống test rỗng luôn xanh nếu enum bị đổi."""
    assert len(list(TranslationStageKind)) >= 4
