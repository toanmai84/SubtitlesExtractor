"""[v3.23.600] Phân tích + dịch có kèm video phải cho NGƯỜI NÓI ở mọi dòng có lời.

Yêu cầu người dùng 10/10/2026: khâu dịch phải bảo đảm đủ 100% từ phân tích tới dịch khi có
ngữ cảnh hình ảnh/âm thanh, để lồng tiếng có đủ 100% người nói. Lượt dịch mới nhất của hai
phim thật (v3.23.583):

* "he thong sung the" ``069d0c24``: nhãn phủ 97,7% nên khâu dịch KHÔNG quét bổ sung (chỉ
  quét khi dưới 60%) — 125/5540 dòng không có người nói. Chín dòng 1301-1313 nằm giữa cuộc
  gọi Lâm Dật ↔ chị dâu là các dòng lưới "lạc cảnh" (v3.23.581) đã bỏ nhãn rồi để trống.
* "傻子落水…" ``da77ec5a``: 103/3902 dòng; ``bf8d54a5`` có 39 dòng chỉ ghi người NGHE
  (dòng 1128 "哎呀，你怎么知道的？" → 柳慕云) mà vẫn được tính là "đã có nhãn".
* Prompt phân tích và bộ quét dặn "không chắc thì để trống người nói".
"""  # noqa: RUF002 — dấu câu chữ Hán trích nguyên văn

from __future__ import annotations

import json
from dataclasses import replace

from subtitles_extractor.application.services.cue_completion import (
    complete_speakers,
    drop_contradicted_fills,
    fill_missing_speakers,
    lines_missing_speaker,
    speaker_coverage,
)
from subtitles_extractor.application.services.cue_contradiction_guard import (
    contradicted_cue_lines,
)
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonical_label,
    canonicalize_cue_labels,
)
from subtitles_extractor.application.services.cue_run_filler import LineSpan
from subtitles_extractor.application.services.dubbing_cast import build_cast
from subtitles_extractor.application.services.speaker_review import (
    REVIEW_VOICE_ONLY,
    SpeakerIssue,
    review_speakers,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
    AnalyzeSubtitleContextUseCase,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesRequest,
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    ANALYSIS_STOP_DAILY_QUOTA,
    SubtitleContextAnalysis,
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.speaker_label import (
    CROWD_SPEAKER,
    UNKNOWN_FEMALE_SPEAKER,
    UNKNOWN_MALE_SPEAKER,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    _speaker_fallback_rule,
    _visual_cues_system_instruction,
)

ROSTER = "\n".join([
    "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính",
    "宋欣爽 / 小爽 (Tống Hân Sảng / Tiểu Sảng) [nữ, ~27] — chị dâu của Lâm Dật",
    "付金奎 (Phó Kim Khuê) [nam, ~50] — phó giám đốc xưởng",
    "钟俊涛 (Chung Tuấn Đào) [nam, ~26] — thiếu gia",
    "王大福 / 胖子 / 三胖子 (Vương Đại Phúc / Bàn Tử) [nam, ~24] — bạn thân Lâm Dật",
])
LIN, SANG = "林逸", "宋欣爽 / 小爽"
FU, ZHONG, DAFU = "付金奎", "钟俊涛", "王大福 / 胖子 / 三胖子"
#: Lượt ``069d0c24`` dòng 1298-1315: (dòng, câu gốc, bắt đầu, kết thúc, người nói, nghe).
ROWS = [
    (1298, "你除了睡觉", 2449600, 2450133, FU, ZHONG),
    (1299, "你还会干嘛", 2450200, 2451000, FU, ZHONG),
    (1300, "得把这个消息", 2451867, 2452867, LIN, SANG),
    (1301, "告诉嫂子啊", 2452933, 2454333, "", ""),
    (1302, "小逸", 2460067, 2460800, DAFU, LIN),
    (1303, "嫂子", 2462133, 2462800, LIN, SANG),
    (1304, "你在医院吗", 2463000, 2464133, "", ""),
    (1305, "我有个好消息", 2464200, 2464733, "", ""),
    (1306, "要告诉你", 2464800, 2465533, "", ""),
    (1307, "我来找你", 2465800, 2466533, "", ""),
    (1308, "不了", 2467000, 2467533, SANG, LIN),
    (1309, "我今天有点不太舒服", 2467933, 2469333, "", ""),
    (1310, "请假了", 2469733, 2470533, "", ""),
    (1311, "不舒服", 2471000, 2471667, "", ""),
    (1312, "嫂子", 2471733, 2472600, LIN, SANG),
    (1313, "你怎么了", 2472667, 2473267, "", ""),
    (1314, "其实也没什么", 2473400, 2475000, SANG, LIN),
    (1315, "你不用太担心", 2475200, 2476533, ZHONG, FU),
]
EMPTY = [row[0] for row in ROWS if not row[4]]
#: Nhãn "lạc cảnh" mà bộ quét thật từng trả cho dòng 1304-1307 (v3.23.578).
LOST = {1304, 1305, 1306, 1307}


def _lines() -> list[TranslationLine]:
    return [
        TranslationLine(index=r[0], start_ms=r[2], end_ms=r[3], text=r[1]) for r in ROWS
    ]


def _cues() -> list[VisualCue]:
    return [VisualCue(r[0], r[4], r[5], "") for r in ROWS if r[4] or r[5]]


class _Scanner:
    """Bộ quét giả: trả nhãn theo ``answers``; ghi lại các dòng được hỏi."""

    def __init__(self, answers: dict[int, tuple[str, str]]) -> None:
        self.answers = answers
        self.asked: list[list[int]] = []

    def __call__(
        self, lines: list[TranslationLine], _known: list[VisualCue]
    ) -> list[VisualCue]:
        self.asked.append([line.index for line in lines])
        return [
            VisualCue(line.index, *self.answers[line.index], "mô tả")
            for line in lines if line.index in self.answers
        ]


# ── Hàm thuần ────────────────────────────────────────────────────────────────


class TestCoverage:
    def test_real_gaps_are_found(self) -> None:
        coverage = speaker_coverage(_cues(), _lines())
        assert coverage.missing == tuple(EMPTY)
        assert coverage.with_speaker == len(ROWS) - len(EMPTY)
        assert [line.index for line in lines_missing_speaker(_cues(), _lines())] == EMPTY

    def test_listener_only_line_has_no_speaker(self) -> None:
        """``bf8d54a5`` dòng 1128: chỉ có người nghe 柳慕云 — lồng tiếng vẫn không biết ai
        nói."""
        line = TranslationLine(1128, 0, 900, "哎呀，你怎么知道的？")  # noqa: RUF001
        cue = VisualCue(1128, "", "柳慕云", "Bác sĩ điện thoại xác nhận bệnh tình")
        assert lines_missing_speaker([cue], [line]) == [line]
        filled, lines = fill_missing_speakers([cue], [VisualCue(1128, "医生", "", "")])
        assert lines == [1128]
        assert filled[0] == replace(cue, speaker="医生")

    def test_empty_text_needs_no_speaker(self) -> None:
        assert lines_missing_speaker([], [TranslationLine(1, 0, 900, "  ")]) == []

    def test_existing_speakers_are_never_replaced(self) -> None:
        filled, lines = fill_missing_speakers(
            _cues(), [VisualCue(1300, FU, ZHONG, ""), VisualCue(1301, LIN, SANG, "")]
        )
        speakers = {cue.line_no: cue.speaker for cue in filled}
        assert lines == [1301]
        assert speakers[1300] == LIN


class TestCompletion:
    def test_second_pass_asks_only_what_is_still_missing(self) -> None:
        first = {line: (LIN, SANG) for line in EMPTY[:5]}
        scanner = _Scanner(first)
        cues, filled = complete_speakers(_cues(), _lines(), scanner)
        assert scanner.asked == [EMPTY, EMPTY[5:]]
        assert filled == EMPTY[:5]
        scanner.answers.update({line: (SANG, LIN) for line in EMPTY[5:]})
        cues, filled = complete_speakers(cues, _lines(), scanner)
        assert speaker_coverage(cues, _lines()).missing == ()

    def test_lost_scene_answers_are_dropped_but_voice_labels_kept(self) -> None:
        """Hỏi lại 1304-1307 mà model lặp nhãn lạc cảnh 付金奎 → 王大福 thì bỏ; dòng
        model chỉ nghe được giọng nữ thì giữ (nhãn giọng không khẳng định danh tính)."""
        answers = {line: (FU, DAFU) for line in LOST}
        answers[1313] = (UNKNOWN_FEMALE_SPEAKER, "")
        original = canonicalize_cue_labels(_cues(), ROSTER)
        completed, filled = complete_speakers(original, _lines(), _Scanner(answers))
        kept, dropped = drop_contradicted_fills(
            original, completed, filled, _lines(), ROSTER
        )
        assert dropped == sorted(LOST)
        speakers = {cue.line_no: cue.speaker for cue in kept}
        assert speakers[1313] == UNKNOWN_FEMALE_SPEAKER
        assert all(line not in speakers for line in LOST)

    def test_voice_label_with_a_listener_is_still_checked(self) -> None:
        cues = canonicalize_cue_labels(
            [*_cues(), VisualCue(1305, UNKNOWN_MALE_SPEAKER, FU, "mô tả")], ROSTER
        )
        spans = {r[0]: LineSpan(r[0], r[2], r[3]) for r in ROWS}
        assert contradicted_cue_lines(cues, {1305}, spans) == [1305]
        bare = [*_cues(), VisualCue(1305, UNKNOWN_MALE_SPEAKER, "", "mô tả")]
        assert contradicted_cue_lines(bare, {1305}, spans) == []


class TestFallbackLabels:
    def test_prompts_ask_for_a_voice_label_not_a_blank(self) -> None:
        rule = _speaker_fallback_rule()
        assert UNKNOWN_MALE_SPEAKER in rule and CROWD_SPEAKER in rule
        scanner = _visual_cues_system_instruction("Vietnamese", ROSTER)
        assert rule in scanner
        assert "để trống thay vì đoán" not in scanner

    def test_spelling_variants_become_one_role(self) -> None:
        for spelling in ("giọng nam", "Giọng nam (chưa rõ)", "GIỌNG NAM CHƯA RÕ"):
            assert canonical_label(spelling, {}) == UNKNOWN_MALE_SPEAKER

    def test_dubbing_reads_the_voice_gender(self) -> None:
        cast = {
            m.name: m.gender
            for m in build_cast([UNKNOWN_MALE_SPEAKER, UNKNOWN_FEMALE_SPEAKER], "")
        }
        assert cast == {
            UNKNOWN_MALE_SPEAKER: VOICE_MALE, UNKNOWN_FEMALE_SPEAKER: VOICE_FEMALE
        }

    def test_voice_only_lines_are_listed_for_review(self) -> None:
        issues = review_speakers(
            [UNKNOWN_FEMALE_SPEAKER, CROWD_SPEAKER], ["你怎么了", "好"], ROSTER, {}
        )
        assert issues == {0: SpeakerIssue(REVIEW_VOICE_ONLY, UNKNOWN_FEMALE_SPEAKER)}


# ── Khâu dịch ────────────────────────────────────────────────────────────────


class _Translator:
    """Translator giả: ``analyze_visual_cues`` trả nhãn theo ``answers``."""

    def __init__(self, answers: dict[int, tuple[str, str]]) -> None:
        self.scanner = _Scanner(answers)

    def analyze_visual_cues(self, source_lines, _target_lang, **_kwargs):
        return self.scanner(list(source_lines), [])

    def is_available(self) -> bool:
        return True


def _request(cues: list[VisualCue]) -> TranslateSubtitlesRequest:
    return TranslateSubtitlesRequest(
        events=[],
        stages=[
            TranslationStageConfig(kind=TranslationStageKind.LITERAL, model_name="m")
        ],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER,
            visual_cues=serialize_visual_cues(cues),
        ),
        video_refs=(object(),),
        enable_visual_cues=True,
    )


class _Checkpoint:
    def __init__(self) -> None:
        self.saved: list[dict] | None = None

    def get_visual_cues(self) -> list[dict] | None:
        return self.saved

    def save_visual_cues(self, cues_json: list[dict]) -> None:
        self.saved = cues_json


class TestTranslation:
    def test_high_coverage_still_asks_for_the_gaps(self) -> None:
        """Phủ 97,7% (> 60%) mà bản trước không hỏi dòng nào."""
        answers = {line: (LIN, SANG) for line in EMPTY}
        translator = _Translator(answers)
        use_case = TranslateSubtitlesUseCase(translator)
        checkpoint = _Checkpoint()
        request = use_case._complete_speaker_cues(
            _lines(), _request(_cues()), checkpoint, None
        )
        assert translator.scanner.asked == [EMPTY]
        speakers = {
            cue.line_no: cue.speaker
            for cue in deserialize_visual_cues(request.context.visual_cues)
        }
        assert all(speakers[line] == LIN for line in EMPTY)
        assert {item["id"] for item in checkpoint.saved} == set(EMPTY)

        # Dịch lại: dùng nhãn đã hỏi trong checkpoint, không hỏi model lần nữa.
        again = _Translator({})
        TranslateSubtitlesUseCase(again)._complete_speaker_cues(
            _lines(), _request(_cues()), checkpoint, None
        )
        assert again.scanner.asked == []

    def test_full_coverage_asks_nothing(self) -> None:
        full = [VisualCue(r[0], r[4] or LIN, r[5], "") for r in ROWS]
        translator = _Translator({})
        request = _request(full)
        assert TranslateSubtitlesUseCase(translator)._complete_speaker_cues(
            _lines(), request, None, None
        ) is request
        assert translator.scanner.asked == []


# ── Khâu phân tích ───────────────────────────────────────────────────────────


class _Analyzer(_Translator):
    def __init__(self, result: SubtitleContextAnalysis, answers) -> None:
        super().__init__(answers)
        self.result = result

    def analyze_global_context(self, **_kwargs) -> SubtitleContextAnalysis:
        return self.result


def _analysis(stop_reason: str = "") -> SubtitleContextAnalysis:
    return SubtitleContextAnalysis(
        source_lang="zh", characters=ROSTER, overview="x",
        visual_cues=json.dumps(
            [{"id": c.line_no, "spk": c.speaker, "to": c.addressee} for c in _cues()],
            ensure_ascii=False,
        ),
        stop_reason=stop_reason,
    )


class TestAnalysis:
    def test_analysis_fills_its_own_gaps(self) -> None:
        answers = {line: (SANG, LIN) for line in EMPTY}
        analyzer = _Analyzer(_analysis(), answers)
        result = AnalyzeSubtitleContextUseCase(analyzer).execute(
            _lines(), "Vietnamese", model_name="m", video_refs=[object()],
            enable_visual_cues=True,
        )
        cues = deserialize_visual_cues(result.visual_cues)
        assert speaker_coverage(cues, _lines()).missing == ()
        assert analyzer.scanner.asked == [EMPTY]

    def test_stopped_analysis_is_not_extended(self) -> None:
        analyzer = _Analyzer(_analysis(ANALYSIS_STOP_DAILY_QUOTA), {})
        result = AnalyzeSubtitleContextUseCase(analyzer).execute(
            _lines(), "Vietnamese", model_name="m", video_refs=[object()],
            enable_visual_cues=True,
        )
        assert analyzer.scanner.asked == []
        assert speaker_coverage(
            deserialize_visual_cues(result.visual_cues), _lines()
        ).missing == tuple(EMPTY)

    def test_without_video_nothing_is_asked(self) -> None:
        analyzer = _Analyzer(_analysis(), {})
        AnalyzeSubtitleContextUseCase(analyzer).execute(
            _lines(), "Vietnamese", model_name="m", video_refs=None,
            enable_visual_cues=True,
        )
        assert analyzer.scanner.asked == []


class _ExecTranslator(_Translator):
    def translate_stage(self, *, input_lines, **_kwargs) -> list[TranslationLine]:
        return [replace(line, text=f"dịch {line.index}") for line in input_lines]


def test_translation_run_asks_for_the_missing_line() -> None:
    """Cả lượt dịch: vùng nhãn dày (lưới điền lượt lời không đụng tới) còn trống dòng
    1313 "你怎么了" → hỏi lại đúng dòng đó, nhãn mới đi vào kết quả dịch."""
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    # Cue ``id`` = vị trí dòng (từ 1) như khâu phân tích thật; dòng 1313 ở vị trí 16.
    position = [r[0] for r in ROWS].index(1313) + 1
    labelled = [
        VisualCue(i, r[4] or LIN, r[5] or SANG, "")
        for i, r in enumerate(ROWS, 1) if i != position
    ]
    events = [
        SubtitleEvent(
            index=i, text=r[1], interval=TimeInterval(r[2] / 1000, r[3] / 1000)
        )
        for i, r in enumerate(ROWS, 1)
    ]
    translator = _ExecTranslator({position: (SANG, LIN)})
    response = TranslateSubtitlesUseCase(translator=translator).execute(
        replace(_request(labelled), events=events)
    )
    assert translator.scanner.asked == [[position]]
    speakers = {
        cue.line_no: cue.speaker for cue in deserialize_visual_cues(response.visual_cues)
    }
    assert speakers[position] == SANG


def test_context_diagnostics_report_speaker_gaps() -> None:
    from subtitles_extractor.application.services.context_diagnostics import (
        build_context_diagnostics,
    )
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    events = [
        SubtitleEvent(
            index=r[0], text=r[1], interval=TimeInterval(r[2] / 1000, r[3] / 1000)
        )
        for r in ROWS
    ]
    context = TranslationContext(
        target_lang="Vietnamese", characters=ROSTER,
        visual_cues=serialize_visual_cues([
            *_cues(), VisualCue(1313, UNKNOWN_FEMALE_SPEAKER, "", "")
        ]),
    )
    report = build_context_diagnostics(context=context, source_events=events)
    cues = report["visual_cues"]
    assert cues["speaker_lines"] == len(ROWS) - len(EMPTY) + 1
    assert cues["voice_only_speaker_lines"] == 1
    assert cues["missing_speaker_lines"] == [line for line in EMPTY if line != 1313]
    codes = {flag["code"] for flag in report["quality_flags"]}
    assert "speakers_missing" in codes


def test_translate_page_reports_speaker_coverage() -> None:
    import pytest

    pytest.importorskip("PySide6")
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    view_model = TranslatePageViewModel.__new__(TranslatePageViewModel)
    view_model._source_events = [
        SubtitleEvent(
            index=i, text=r[1], interval=TimeInterval(r[2] / 1000, r[3] / 1000)
        )
        for i, r in enumerate(ROWS, 1)
    ]
    cues = [VisualCue(i, r[4], r[5], "") for i, r in enumerate(ROWS, 1) if r[4]]
    message = view_model.speaker_coverage_message(serialize_visual_cues(cues))
    assert message == (
        f"Người nói: {len(cues)}/{len(ROWS)} dòng có lời (50.0%) — còn "
        f"{len(EMPTY)} dòng chưa có người nói."
    )
    assert view_model.speaker_coverage_message("") == ""
