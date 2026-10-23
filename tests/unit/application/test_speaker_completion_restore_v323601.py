# ruff: noqa: RUF001, RUF002 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.601] "Bổ sung người nói" cho phân tích KHÔI PHỤC từ phiên cũ.

Chuẩn đoán thật 灭门惨案II (v3.23.600, 18:05): mở lại video → phân tích được khôi phục từ
phiên cũ (không qua bước hỏi lại người nói của khâu phân tích/dịch) → sang TTS ngay. Kết
quả: 39/1246 dòng không có người nói, đọc bằng giọng chung "Minh Đức" — có cả dòng 714
"不要！" của Tần Khả Tú (hai dòng trước, một dòng sau đều là cô ấy). Các dòng dưới đây là
dữ liệu thật của phiên đó (mốc ms, nhãn spk/to của phân tích khôi phục).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.application.use_cases.complete_speakers import (
    CompleteSpeakersUseCase,
    SpeakerCompletionResult,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleTranslationError,
    TranslationCancelledError,
    TranslationLine,
    VisualCue,
)

# (dòng, câu gốc, bắt đầu ms, kết thúc ms, người nói, người nghe)
ROWS = [
    (100, "隊長，我可以的，要不要裝滿？", 480467, 485400, "老伯", "龍傳人"),
    (101, "我怕我受.....受不了呀隊長", 485467, 488467, "老伯", "龍傳人"),
    (102, "好榜樣！", 491133, 492133, "龍傳人", "老伯"),
    (103, "怎交貨呀", 492200, 493400, "", "龍傳人"),
    (104, "又沒有女人，甚麼都沒有", 493467, 494533, "", "龍傳人"),
    (105, "突然間怎麼交", 494600, 495933, "", "龍傳人"),
    (106, "别吵", 497267, 498400, "龍傳人", ""),
    (107, "誰不交貨，證明誰就是做了虧心事", 501333, 504333, "龍傳人", ""),
    (108, "不是呢，所長，你示範一次", 504400, 506667, "", "龍傳人"),
    (109, "我們大家不就是跟著做嘛", 506733, 508067, "", "龍傳人"),
    (110, "對呀，對！", 508133, 511267, "", ""),
    (111, "不然我們可不知道怎麼交貨", 512400, 514733, "", "龍傳人"),
    (112, "女人照片都沒一張", 514800, 516200, "", "龍傳人"),
    (113, "我們怎麼做呢", 516267, 518200, "", "龍傳人"),
    (114, "怎麼交貨呀", 518267, 520200, "", "龍傳人"),
    (115, "愛國，既然你如此愛國，就讓你.......", 522606, 525637, "龍傳人", "劉愛國"),
    (710, "但我真的喜歡你", 3314467, 3315667, "朱正軍", "秦可秀"),
    (711, "不要呀不要！", 3316000, 3316933, "秦可秀", "朱正軍"),
    (712, "不要.....不要！", 3317600, 3321333, "秦可秀", "朱正軍"),
    (713, "不要！", 3323600, 3324667, "秦可秀", "朱正軍"),
    (714, "不要！", 3325333, 3326267, "", ""),
    (715, "放開我呀", 3356267, 3357733, "秦可秀", "朱正軍"),
]
ROSTER = (
    "龍傳人 (Long Truyền Nhân) [nam, ~35] — Trưởng đồn công an\n"
    "老伯 (Lão bá) [nam, ~65] — ông lão trong làng\n"
    "劉愛國 / 愛國 (Lưu Ái Quốc / Ái Quốc) [nam, ~40] — dân làng\n"
    "秦可秀 / 阿秀 (Tần Khả Tú / A Tú) [nữ, ~28] — vợ Chân Kiếm Thanh\n"
    "朱正軍 / 阿軍 (Chu Chính Quân / A Quân) [nam, ~30] — cựu chiến binh"
)
MISSING = [row[0] for row in ROWS if not row[4]]
#: Câu trả lời của model cho các dòng thiếu (dân làng nói với trưởng đồn; 714 là cô Tú).
ANSWERS = {line: "Đám đông" for line in MISSING if line != 714} | {714: "秦可秀"}


def _lines() -> list[TranslationLine]:
    return [
        TranslationLine(index=row[0], start_ms=row[2], end_ms=row[3], text=row[1])
        for row in ROWS
    ]


def _cues_json() -> str:
    return serialize_visual_cues(
        [VisualCue(row[0], row[4], row[5], "") for row in ROWS if row[4] or row[5]]
    )


class _Translator:
    """Bộ dịch giả: ghi lại các dòng được hỏi, trả lời theo ``answers``."""

    def __init__(
        self,
        answers: dict[int, str],
        error: Exception | None = None,
        fail_calls: int = 0,
    ) -> None:
        self.answers = answers
        self.error = error
        self.fail_calls = fail_calls
        self.asked: list[list[int]] = []
        self.video_refs: list[object] = []

    def analyze_visual_cues(self, lines, target_lang, **kwargs) -> list[VisualCue]:
        self.asked.append([line.index for line in lines])
        self.video_refs = kwargs["video_refs"]
        if self.error is not None and len(self.asked) <= self.fail_calls:
            raise self.error
        return [
            VisualCue(line.index, self.answers[line.index], "", "")
            for line in lines
            if line.index in self.answers
        ]


def _execute(translator: _Translator, **overrides) -> SpeakerCompletionResult:
    arguments = {
        "characters": ROSTER, "target_lang": "Vietnamese",
        "model_name": "gemini-3.1-flash-lite", "video_refs": ["chunk-1"],
    } | overrides
    return CompleteSpeakersUseCase(translator).execute(
        _lines(), _cues_json(), **arguments
    )


def test_chi_hoi_dung_cac_dong_thieu_va_giu_nhan_cu() -> None:
    translator = _Translator(ANSWERS)
    result = _execute(translator)
    assert translator.asked == [MISSING]
    assert translator.video_refs == ["chunk-1"]
    assert (result.before.with_speaker, result.after.with_speaker) == (11, 22)
    assert result.changed and result.after.missing == ()
    by_line = {cue.line_no: cue for cue in deserialize_visual_cues(result.visual_cues)}
    for row in ROWS:
        if row[4]:
            # Nhãn sẵn có giữ nguyên người (chỉ quy về tên đầy đủ trong bảng nhân vật).
            assert by_line[row[0]].speaker.split(" / ")[0] == row[4]
    # Giọng nữ của cô Tú, không còn giọng chung.
    assert by_line[714].speaker == "秦可秀 / 阿秀"
    assert by_line[103].speaker == "Đám đông"
    assert by_line[103].addressee == "龍傳人"  # người nghe cũ không bị xoá


def test_luot_hai_chi_hoi_dong_model_con_bo_sot() -> None:
    first_half = {line: ANSWERS[line] for line in MISSING[:5]}

    class _TwoPass(_Translator):
        def analyze_visual_cues(self, lines, target_lang, **kwargs):
            self.answers = first_half if not self.asked else ANSWERS
            return super().analyze_visual_cues(lines, target_lang, **kwargs)

    translator = _TwoPass({})
    result = _execute(translator)
    assert translator.asked == [MISSING, MISSING[5:]]
    assert result.after.missing == ()


def test_moi_luot_deu_loi_thi_bao_loi_cho_nguoi_dung() -> None:
    quota = RuntimeError("Hết hạn mức ngày cho gemini-3.1-flash-lite")
    translator = _Translator(ANSWERS, error=quota, fail_calls=99)
    with pytest.raises(RuntimeError, match="Hết hạn mức"):
        _execute(translator)
    assert len(translator.asked) == 1  # không có gì mới → không hỏi lượt hai


def test_luot_sau_loi_van_giu_phan_da_dien() -> None:
    first_half = {line: ANSWERS[line] for line in MISSING[:5]}

    class _SecondPassFails(_Translator):
        def analyze_visual_cues(self, lines, target_lang, **kwargs):
            if self.asked:
                self.asked.append([line.index for line in lines])
                raise OSError("mạng chập chờn")
            return super().analyze_visual_cues(lines, target_lang, **kwargs)

    translator = _SecondPassFails(first_half)
    result = _execute(translator)
    assert len(translator.asked) == 2
    assert result.changed
    assert result.after.missing == tuple(MISSING[5:])


def test_nguoi_dung_huy_thi_dung_han() -> None:
    translator = _Translator(
        ANSWERS, error=TranslationCancelledError("huỷ"), fail_calls=99
    )
    with pytest.raises(TranslationCancelledError):
        _execute(translator, cancel_cb=lambda: True)


def test_bao_huy_khong_do_nguoi_dung_la_loi_thuong() -> None:
    translator = _Translator(
        ANSWERS, error=TranslationCancelledError("timeout"), fail_calls=99
    )
    with pytest.raises(TranslationCancelledError):
        _execute(translator, cancel_cb=lambda: False)
    assert len(translator.asked) == 1


@pytest.mark.parametrize(
    ("overrides", "visual_cues"),
    [({"video_refs": []}, None), ({}, "")],
)
def test_thieu_video_hoac_cues_thi_bao_ro(
    overrides: dict, visual_cues: str | None
) -> None:
    translator = _Translator(ANSWERS)
    arguments = {
        "characters": ROSTER, "target_lang": "Vietnamese",
        "model_name": "gemini-3.1-flash-lite", "video_refs": ["chunk-1"],
    } | overrides
    with pytest.raises(SubtitleTranslationError):
        CompleteSpeakersUseCase(translator).execute(
            _lines(), _cues_json() if visual_cues is None else visual_cues, **arguments
        )
    assert translator.asked == []


def test_khau_phan_tich_dung_chung_loi() -> None:
    from subtitles_extractor.application.use_cases.analyze_subtitle_context import (
        AnalyzeSubtitleContextUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        SubtitleContextAnalysis,
    )

    translator = _Translator(ANSWERS)
    analysis = SubtitleContextAnalysis(characters=ROSTER, visual_cues=_cues_json())
    completed = AnalyzeSubtitleContextUseCase(translator)._complete_speakers(
        analysis, _lines(), "Vietnamese", "gemini-3.1-flash-lite", ["chunk-1"],
        150, None, None,
    )
    assert translator.asked == [MISSING]
    filled = {cue.line_no for cue in deserialize_visual_cues(completed.visual_cues)}
    assert set(MISSING) <= filled


# ── Trang Dịch ───────────────────────────────────────────────────────────────


def _view_model():
    pytest.importorskip("PySide6")
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    view_model = TranslatePageViewModel.__new__(TranslatePageViewModel)
    view_model._source_events = [
        SubtitleEvent(
            index=row[0], text=row[1], interval=TimeInterval(row[2] / 1000, row[3] / 1000)
        )
        for row in ROWS
    ]
    return view_model


def test_dem_dong_thieu_nguoi_noi() -> None:
    view_model = _view_model()
    assert view_model.missing_speaker_count(_cues_json()) == len(MISSING)
    assert view_model.missing_speaker_count("") == 0


class _Signal:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, value: object) -> None:
        self.emitted.append(value)


def _guarded_view_model(busy: bool = False):
    view_model = _view_model()
    view_model._is_busy = busy
    view_model.status_message = _Signal()
    view_model.error_occurred = _Signal()
    view_model.analyze_error = _Signal()
    return view_model


def test_nut_can_video() -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    view_model = _guarded_view_model()
    started = TranslatePageViewModel.start_speaker_completion(
        view_model, "key", "gemini-3.1-flash-lite", None, _cues_json(), ROSTER,
        "Vietnamese",
    )
    assert started is False
    assert "video" in view_model.analyze_error.emitted[0]


def test_nut_khong_chay_khi_da_du_nguoi_noi() -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    full = serialize_visual_cues(
        [VisualCue(row[0], row[4] or "Đám đông", row[5], "") for row in ROWS]
    )
    view_model = _guarded_view_model()
    assert TranslatePageViewModel.start_speaker_completion(
        view_model, "key", "gemini-3.1-flash-lite", "/phim.mp4", full, ROSTER,
        "Vietnamese",
    ) is False
    assert view_model.status_message.emitted == ["Mọi dòng có lời đã có người nói."]


def test_ket_qua_duoc_luu_vao_phien_cua_video_va_bao_lai() -> None:
    from subtitles_extractor.presentation.view_models.translate_page_view_model import (
        TranslatePageViewModel,
    )

    result = _execute(_Translator(ANSWERS))
    saved: list[tuple[str, str]] = []
    view_model = SimpleNamespace(
        _speaker_completion_video_path="F:/灭门惨案II.mp4",
        _persist_topped_up_cues=lambda cues, path: saved.append((cues, path)),
        speaker_completion_done=_Signal(),
        _close_token_report=lambda: "",  # [v3.23.610] báo token của lượt hỏi lại
    )
    TranslatePageViewModel._on_speaker_completion_finished(view_model, result)
    assert saved == [(result.visual_cues, "F:/灭门惨案II.mp4")]
    assert view_model.speaker_completion_done.emitted == [
        "Người nói: 11 → 22/22 dòng có lời."
    ]


class _Button:
    def __init__(self) -> None:
        self.text = ""
        self.tooltip = ""
        self.enabled: bool | None = None

    def setText(self, text: str) -> None:  # noqa: N802 (giả lập Qt)
        self.text = text

    def setToolTip(self, text: str) -> None:  # noqa: N802
        self.tooltip = text

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        self.enabled = enabled


def _page(visual_cues: str, busy: bool = False) -> SimpleNamespace:
    from pathlib import Path

    from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator

    data_dir = Path("src/subtitles_extractor/data")
    view_model = _view_model()
    view_model._is_busy = busy
    return SimpleNamespace(
        _btn_complete_speakers=_Button(),
        _visual_cues_edit=SimpleNamespace(toPlainText=lambda: visual_cues),
        _view_model=SimpleNamespace(
            missing_speaker_count=view_model.missing_speaker_count, is_busy=busy
        ),
        _translator=JsonTranslator(data_dir=data_dir, default_locale="vi"),
    )


def test_nut_hien_so_dong_thieu() -> None:
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    page = _page(_cues_json())
    TranslatePage._update_speaker_button(page)
    assert page._btn_complete_speakers.text == f"🎭 Bổ sung người nói ({len(MISSING)})"
    assert page._btn_complete_speakers.enabled is True

    busy = _page(_cues_json(), busy=True)
    TranslatePage._update_speaker_button(busy)
    assert busy._btn_complete_speakers.enabled is False

    empty = _page("")
    TranslatePage._update_speaker_button(empty)
    assert empty._btn_complete_speakers.text == "🎭 Bổ sung người nói"
    assert empty._btn_complete_speakers.enabled is False


def test_khoi_phuc_phan_tich_thieu_nguoi_noi_thi_chi_cach_bo_sung() -> None:
    from pathlib import Path

    from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
    from subtitles_extractor.presentation.pages.translate_page import TranslatePage

    warnings: list[tuple[str, str]] = []
    view_model = _view_model()
    page = SimpleNamespace(
        _fill_context_fields=lambda _result: None,
        _show_info=lambda _title, _body: None,
        _show_warning=lambda title, body: warnings.append((title, body)),
        _update_action_states=lambda: None,
        _translator=JsonTranslator(
            data_dir=Path("src/subtitles_extractor/data"), default_locale="vi"
        ),
        _view_model=SimpleNamespace(
            stage_comparison=lambda: [],
            stage_output_count=0,  # [v3.23.609] trang đếm giai đoạn không dựng lại chữ
            missing_speaker_count=view_model.missing_speaker_count,
            roster_without_address_rules=lambda _characters: 0,
        ),
    )
    restored = SimpleNamespace(characters=ROSTER, visual_cues=_cues_json())
    TranslatePage._on_analysis_restored(page, restored)
    assert len(warnings) == 1
    title, body = warnings[0]
    assert title == "Bổ sung người nói"
    assert body.startswith(f"{len(MISSING)} dòng có lời chưa có người nói")
    assert "🎭 Bổ sung người nói" in body
