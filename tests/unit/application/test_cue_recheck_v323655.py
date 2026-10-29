"""[v3.23.655] Phân tích lại CÓ CHỌN LỌC: chỉ hỏi lại dòng bị chấm nghi; nhãn mới chỉ nhận khi hết nghi; cue người dùng
không bị đụng."""

from __future__ import annotations

import pytest

from subtitles_extractor.application.services.visual_cue_serializer import deserialize_visual_cues
from subtitles_extractor.application.use_cases.complete_speakers import RecheckSuspiciousCuesUseCase
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationLine,
    VisualCue,
)

ROSTER = "林逸 (Lâm Dật) [nam, ~20] — chính\n苏瑶 (Tô Dao) [nữ, ~18] — sư muội"
TEXTS = ["师妹你来了", "本姑娘不怕", "林逸你走吧", "好", "我来了"]
CUES = (
    '[{"id":1,"spk":"林逸","to":"苏瑶"},'
    '{"id":2,"spk":"林逸","to":"苏瑶"},'           # 本姑娘 mà người nói là nam → nghi
    '{"id":3,"spk":"林逸","to":"苏瑶"},'           # câu gọi chính "林逸" → nghi
    '{"id":4,"spk":"苏瑶","to":"苏瑶","src":"user"},'  # người dùng chọn (dù người nói = người nghe)
    '{"id":5,"spk":"苏瑶","to":"林逸"}]'
)


def _lines() -> list[TranslationLine]:
    return [TranslationLine(index=i + 1, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=t, original_text=t)
            for i, t in enumerate(TEXTS)]


class _Translator:
    def __init__(self, answers: dict[int, tuple[str, str]]) -> None:
        self.answers = answers
        self.asked: list[list[int]] = []
        self.known: list[int] = []

    def analyze_visual_cues(self, lines, target_lang, *, known_cues=None, **_kw):
        self.asked.append([line.index for line in lines])
        self.known = [cue.line_no for cue in known_cues or []]
        return [VisualCue(i, speaker=s, addressee=a) for i, (s, a) in self.answers.items()]


def _run(translator) -> object:
    return RecheckSuspiciousCuesUseCase(translator).execute(
        _lines(), CUES, characters=ROSTER, target_lang="vi", model_name="m", video_refs=[object()],
    )


def test_only_suspicious_lines_are_asked_and_fixed() -> None:
    translator = _Translator({2: ("苏瑶", "林逸"), 3: ("苏瑶", "林逸"), 4: ("林逸", "苏瑶")})
    result = _run(translator)
    assert translator.asked == [[2, 3]]                       # không hỏi dòng ổn / dòng người dùng
    assert 2 not in translator.known and 1 in translator.known  # nhãn dòng quanh làm ngữ cảnh
    cues = {c.line_no: c for c in deserialize_visual_cues(result.visual_cues)}
    assert (cues[2].speaker, cues[3].speaker) == ("苏瑶", "苏瑶")
    assert cues[4].speaker == "苏瑶" and cues[4].origin == "user"  # không bị đụng
    assert result.fixed == (2, 3) and result.suspicious == 2


def test_new_label_still_suspicious_is_rejected() -> None:
    translator = _Translator({2: ("林逸", "林逸"), 3: ("林逸", "苏瑶")})   # vẫn sai / y như cũ
    result = _run(translator)
    assert result.fixed == () and set(result.unchanged) == {2, 3}
    assert result.visual_cues == CUES


def test_requires_video() -> None:
    with pytest.raises(SubtitleTranslationError):
        RecheckSuspiciousCuesUseCase(_Translator({})).execute(
            _lines(), CUES, characters=ROSTER, target_lang="vi", model_name="m", video_refs=[],
        )
