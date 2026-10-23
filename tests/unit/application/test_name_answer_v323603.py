# ruff: noqa: RUF001, RUF003 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.603] Dòng TRẢ LỜI tên không phải tiếng gọi: người nói là người mang tên.

Chuẩn đoán thật 11/10 (灭门惨案II, gemini-3.8-flash): dòng 126 Long Truyền Nhân hỏi
"你叫甚麼名字", dòng 127 Tần Khả Tú đáp "秦可秀". Lưới đảo nhãn dòng gọi tên (v3.23.570)
coi 127 là tiếng gọi nên đảo nhãn đúng ``秦可秀 → 龍傳人`` thành ``龍傳人 → 秦可秀``, lưu
vào phiên — lồng tiếng đọc "Tần Khả Tú." bằng giọng nam (Minh Đức). Cả 22 lượt phân tích
đã xuất của phim đều mang nhãn sai này. Phim khác (dòng 2442-2444): "如何称呼" / "我是" /
"柳慕云!" — 16 lượt ghi dòng tên là lời 徐俊豪 dù dòng "我是" ngay trước ghi đúng 柳慕云.

Mô phỏng trên 30 lượt phân tích khác nhau: chỉ hai dòng này đổi nhãn (127 ở 11 lượt,
2444 ở 4 lượt), đều thành đúng; mọi dòng gọi tên khác giữ nguyên kết quả cũ.
"""

from __future__ import annotations

from types import SimpleNamespace

from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.application.services.vocative_guard import (
    fix_name_answer_cues,
    fix_vocative_cue_labels,
    is_name_answer,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue

ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~35-40] — Trưởng đồn công an Đông Long; xưng hô: "
    "với Tần Khả Tú tự xưng 'tôi' gọi 'cô'",
    "秦可秀 (Tần Khả Tú) [nữ, ~25] — vợ của Chân Kiếm Thanh; xưng hô: với Long Truyền "
    "Nhân tự xưng 'tôi' gọi 'đồng chí công an'",
    "愛國 (Ái Quốc) [nam, ~25-30] — Công an viên đồn Đông Long",
])
#: Dòng 125-128 thật.
SOURCES = {
    125: "她丈夫甄劍聲撞車死了",
    126: "你叫甚麼名字",
    127: "秦可秀",
    128: "我們現在做大事，這些小事情.",
}
#: Nhãn ĐÃ LƯU của phiên chuẩn đoán (sau lưới đảo nhãn của v3.23.602).
STORED = [
    VisualCue(125, "愛國", "龍傳人"),
    VisualCue(126, "龍傳人", "秦可秀"),
    VisualCue(127, "龍傳人", "秦可秀"),
    VisualCue(128, "龍傳人", "秦可秀"),
]

ROSTER_2 = "\n".join([
    "安雪漫 (An Tuyết Mạn) [nữ, ~23] — vợ mới cưới của Từ Tuấn Hào",
    "徐俊豪 (Từ Tuấn Hào) [nam, ~25] — thiếu gia nhà họ Từ",
    "柳慕云 (Liễu Mộ Vân) [nữ, ~26] — Tổng giám đốc Tập đoàn họ Liễu",
    "沈明月 (Thẩm Minh Nguyệt) [nữ, ~26] — bạn thân của Liễu Mộ Vân",
    "宋钱 (Tống Tiền) [nam, ~25] — thần y",
])
#: Dòng 2442-2445 thật (phim "傻子落水…", lượt 07e94e55).
SOURCES_2 = {2442: "如何称呼", 2443: "我是", 2444: "柳慕云!", 2445: "什么?"}
CUES_2 = [
    VisualCue(2442, "徐俊豪", "柳慕云"),
    VisualCue(2443, "柳慕云", "徐俊豪"),
    VisualCue(2444, "徐俊豪", "安雪漫"),
    VisualCue(2445, "安雪漫", "徐俊豪"),
]


def _labels(cues: list[VisualCue]) -> dict[int, tuple[str, str]]:
    return {cue.line_no: (cue.speaker, cue.addressee) for cue in cues}


def test_nhan_dap_ten_da_luu_duoc_sua_ve_nguoi_mang_ten() -> None:
    fixes = fix_vocative_cue_labels(STORED, SOURCES, ROSTER)
    assert fixes.name_answers == [127]
    assert fixes.swapped == []
    assert _labels(fixes.cues)[127] == ("秦可秀", "龍傳人")
    assert _labels(fixes.cues)[126] == ("龍傳人", "秦可秀")


def test_nhan_dung_cua_model_khong_bi_dao_nua() -> None:
    raw = [*STORED[:2], VisualCue(127, "秦可秀", "龍傳人"), STORED[3]]
    fixes = fix_vocative_cue_labels(raw, SOURCES, ROSTER)
    assert not fixes.changed
    assert _labels(fixes.cues)[127] == ("秦可秀", "龍傳人")


def test_chay_lai_tren_nhan_da_sua_khong_doi_gi() -> None:
    once = fix_vocative_cue_labels(STORED, SOURCES, ROSTER).cues
    assert not fix_vocative_cue_labels(once, SOURCES, ROSTER).changed


def test_tu_gioi_thieu_bi_tach_dong() -> None:
    fixed, changed = fix_name_answer_cues(CUES_2, SOURCES_2, ROSTER_2)
    assert changed == [2444]
    assert _labels(fixed)[2444] == ("柳慕云", "徐俊豪")
    assert is_name_answer(SOURCES_2, 2444)


def test_tieng_goi_that_van_duoc_dao() -> None:
    # Dòng 2531-2532 thật: "我明白了" rồi "柳慕云" — tiếng gọi, nhãn ghi ngược.
    sources = {2531: "我明白了", 2532: "柳慕云"}
    cues = [VisualCue(2531, "柳慕云", "沈明月"), VisualCue(2532, "柳慕云", "沈明月")]
    fixes = fix_vocative_cue_labels(cues, sources, ROSTER_2)
    assert fixes.swapped == [2532]
    assert _labels(fixes.cues)[2532] == ("沈明月", "柳慕云")


def test_hoi_goi_toi_la_gi_khong_phai_cau_hoi_ten() -> None:
    # Dòng 1997-1998 thật: "你叫我什么？" (gọi tôi là gì) rồi "宋钱啊?" của người kia.
    sources = {1997: "你叫我什么？", 1998: "宋钱啊?"}
    assert not is_name_answer(sources, 1998)
    cues = [VisualCue(1997, "宋钱", "徐俊豪"), VisualCue(1998, "宋钱", "徐俊豪")]
    fixes = fix_vocative_cue_labels(cues, sources, ROSTER_2)
    assert fixes.swapped == [1998]


def test_long_tieng_doc_dong_dap_ten_bang_giong_nu() -> None:
    from subtitles_extractor.application.services.dubbing_cast import (
        DubbingSource,
        line_speakers,
    )
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    spans = tuple((line * 2.0, line * 2.0 + 1.5) for line in range(1, 129))
    texts = tuple(SOURCES.get(line, f"câu {line}") for line in range(1, 129))
    source = DubbingSource(ROSTER, serialize_visual_cues(STORED), spans, texts)
    events = [
        SubtitleEvent(index=line, text=f"dòng {line}", interval=TimeInterval(*span))
        for line, span in enumerate(spans, 1)
    ]
    speakers = dict(enumerate(line_speakers(events, source), 1))
    assert speakers[127] == "秦可秀"
    assert speakers[126] == "龍傳人"


def test_chuan_doan_ngu_canh_bao_dong_dap_ten() -> None:
    from subtitles_extractor.application.services.context_diagnostics import (
        _vocative_cue_label_fixes,
    )

    events = [
        SimpleNamespace(index=line, text=SOURCES.get(line, "…"))
        for line in range(1, 129)
    ]
    swapped, kin, answers = _vocative_cue_label_fixes(
        serialize_visual_cues(STORED), events, ROSTER
    )
    assert (swapped, kin, answers) == ([], [], [127])


def test_khau_dich_sua_nhan_truoc_khi_dich() -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
        TranslationLine,
    )

    request = TranslateSubtitlesRequest(
        events=[],
        stages=[],
        context=TranslationContext(
            target_lang="Vietnamese",
            characters=ROSTER,
            visual_cues=serialize_visual_cues(STORED),
        ),
    )
    lines = [
        TranslationLine(index=line, start_ms=line * 1000, end_ms=line * 1000 + 900,
                        text=SOURCES.get(line, "…"))
        for line in range(1, 129)
    ]
    fixed = TranslateSubtitlesUseCase._fix_vocative_cues(request, lines)
    cues = _labels(deserialize_visual_cues(fixed.context.visual_cues))
    assert cues[127] == ("秦可秀", "龍傳人")
