"""[v3.23.602] Nhãn người nói vừa hỏi lại phải có chỗ dựa trong cảnh; model hỏi lại.

Chuẩn đoán thật 灭门惨案II (dịch v3.23.601 bằng gemini-3.5-flash-lite, 11/10 05:38): bước
hỏi lại điền 39/39 dòng thiếu người nói, nhưng chia LUÂN PHIÊN ba tên 胖公安 / 老伯 /
朱正軍 cho đám dân làng (103-114, 131-132, 154-156). Dữ liệu dưới đây là nhãn thật của
phiên đó (dòng 60-165).

Số đo (xem ``fill_support_guard``): người nói thật vắng mặt khỏi nhãn ±10 dòng ở 0.83%
nhãn phân tích (153 746 nhãn, 57 lượt, 3 phim); nhãn tên vừa hỏi lại: 14/34 (41%), không
dòng nào trong 14 dòng đó trùng nhãn đa số của các lượt phân tích khác. Mô phỏng trên nhãn
thật: hạ nhầm 189/31 340 (0.60%).
"""

from __future__ import annotations

import json

import pytest

from subtitles_extractor.application.services.cue_completion import (
    complete_and_check_speakers,
)
from subtitles_extractor.application.services.fill_support_guard import (
    REASON_ABSENT,
    demote_unsupported_fills,
    demote_unsupported_scan_labels,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationLine,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.speaker_label import (
    UNKNOWN_MALE_SPEAKER,
)

ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~35] — Trưởng đồn công an",
    "余小蓮 / 當當 (Dư Tiểu Liên / Đang Đang) [nữ, ~22] — nữ công an",
    "秦可秀 / 阿秀 (Tần Khả Tú / A Tú) [nữ, ~25] — vợ Chân Kiếm Thanh",
    "朱正軍 / 阿軍 (Chu Chính Quân / A Quân) [nam, ~30] — cựu chiến binh",
    "劉愛國 / 愛國 (Lưu Ái Quốc / Ái Quốc) [nam, ~25] — dân làng",
    "胖公安 (Cảnh sát béo) [nam, ~28] — cấp dưới của Long Truyền Nhân",
    "老伯 (Lão ông cầm bô) [nam, ~65] — ông lão trong làng",
    "小孩 (Đứa trẻ) [nam, ~10] — cậu bé trong làng",
])

#: Nhãn phân tích ĐÃ CÓ (người nói, người nghe) dòng 60-165 — phiên 3.23.600.
ORIGINAL = {
    60: ('劉愛國', '胖公安'),
    61: ('劉愛國', '胖公安'),
    62: ('胖公安', '劉愛國'),
    63: ('龍傳人', '劉愛國'),
    64: ('劉愛國', '龍傳人'),
    65: ('劉愛國', '龍傳人'),
    66: ('龍傳人', '劉愛國'),
    67: ('龍傳人', '劉愛國'),
    68: ('劉愛國', '龍傳人'),
    69: ('胖公安', ''),
    75: ('劉愛國', ''),
    76: ('劉愛國', ''),
    77: ('劉愛國', '龍傳人'),
    78: ('龍傳人', ''),
    79: ('龍傳人', ''),
    80: ('', '龍傳人'),
    81: ('龍傳人', ''),
    82: ('龍傳人', ''),
    83: ('', '龍傳人'),
    84: ('龍傳人', ''),
    85: ('小孩', '龍傳人'),
    86: ('小孩', ''),
    87: ('小孩', ''),
    88: ('小孩', ''),
    89: ('小孩', ''),
    90: ('小孩', ''),
    91: ('小孩', ''),
    92: ('龍傳人', ''),
    93: ('龍傳人', ''),
    94: ('龍傳人', ''),
    95: ('龍傳人', '小孩'),
    96: ('小孩', '龍傳人'),
    97: ('小孩', '龍傳人'),
    98: ('劉愛國', '小孩'),
    99: ('小孩', '劉愛國'),
    100: ('老伯', '龍傳人'),
    101: ('老伯', '龍傳人'),
    102: ('龍傳人', '老伯'),
    103: ('', '龍傳人'),
    104: ('', '龍傳人'),
    105: ('', '龍傳人'),
    106: ('龍傳人', ''),
    107: ('龍傳人', ''),
    108: ('', '龍傳人'),
    109: ('', '龍傳人'),
    111: ('', '龍傳人'),
    112: ('', '龍傳人'),
    113: ('', '龍傳人'),
    114: ('', '龍傳人'),
    115: ('龍傳人', '劉愛國'),
    116: ('龍傳人', '劉愛國'),
    117: ('劉愛國', '龍傳人'),
    118: ('龍傳人', '劉愛國'),
    119: ('龍傳人', '秦可秀'),
    120: ('龍傳人', '秦可秀'),
    121: ('秦可秀', '龍傳人'),
    122: ('秦可秀', '龍傳人'),
    123: ('龍傳人', '劉愛國'),
    124: ('劉愛國', '龍傳人'),
    125: ('劉愛國', '龍傳人'),
    126: ('龍傳人', '秦可秀'),
    127: ('龍傳人', '秦可秀'),
    128: ('龍傳人', '秦可秀'),
    129: ('劉愛國', '龍傳人'),
    130: ('劉愛國', '秦可秀'),
    131: ('', '龍傳人'),
    132: ('', '龍傳人'),
    133: ('龍傳人', ''),
    134: ('', '龍傳人'),
    136: ('龍傳人', '余小蓮'),
    137: ('龍傳人', ''),
    138: ('', '龍傳人'),
    139: ('龍傳人', '余小蓮'),
    140: ('余小蓮', '龍傳人'),
    141: ('余小蓮', '龍傳人'),
    142: ('余小蓮', '龍傳人'),
    143: ('龍傳人', '余小蓮'),
    144: ('龍傳人', ''),
    145: ('', '龍傳人'),
    146: ('', '余小蓮'),
    148: ('龍傳人', ''),
    149: ('', '龍傳人'),
    150: ('龍傳人', '胖公安'),
    151: ('龍傳人', '胖公安'),
    152: ('龍傳人', '胖公安'),
    153: ('龍傳人', '余小蓮'),
    157: ('龍傳人', ''),
    158: ('龍傳人', ''),
    159: ('龍傳人', ''),
    160: ('龍傳人', ''),
    161: ('龍傳人', ''),
    162: ('村民', '龍傳人'),
    163: ('村民', '龍傳人'),
    164: ('龍傳人', '村民'),
    165: ('龍傳人', '村民'),
}
#: Nhãn HỎI LẠI của lượt dịch 3.23.601 (gemini-3.5-flash-lite) — (người nói, mô tả).
FILLS = {
    70: ('胖公安', 'Đứng trong hàng chờ, sốt ruột và thắc mắ'),
    71: ('老伯', 'Đứng tuổi cầm bô, phàn nàn về việc đóng '),
    72: ('朱正軍 / 阿軍', "Đứng cạnh lẩm bẩm về việc chia 'thịt'."),
    73: ('龍傳人', 'Đeo kính râm đứng trên bục, hùng hồn phá'),
    74: ('龍傳人', 'Đứng oai phong trên bục giảng giải.'),
    80: ('老伯', 'Thắc mắc về vụ án Hồng Chấn.'),
    83: ('胖公安', "Hỏi ngây ngô về thuật ngữ 'tự sướng'."),
    103: ('胖公安', 'Thắc mắc không có điều kiện để giao mẫu.'),
    104: ('老伯', 'Hùa theo đám đông phàn nàn thiếu phụ nữ.'),
    105: ('朱正軍 / 阿軍', 'Bày tỏ sự khó khăn khi nộp mẫu.'),
    108: ('胖公安', 'Đề nghị trưởng đồn làm mẫu trước.'),
    109: ('老伯', 'Ủng hộ việc trưởng đồn làm mẫu.'),
    110: ('Đám đông', 'Đám đông đồng thanh hưởng ứng.'),
    111: ('朱正軍 / 阿軍', 'Giải thích lý do không biết cách nộp.'),
    112: ('老伯', 'Phàn nàn vì thiếu hình ảnh.'),
    113: ('胖公安', 'Bối rối hỏi cách thực hiện.'),
    114: ('朱正軍 / 阿軍', 'Liên tục giục trưởng đồn hướng dẫn.'),
    131: ('胖公安', 'Nhắc lại lời trưởng đồn hứa hẹn.'),
    132: ('老伯', 'Ủng hộ đòi hỏi có phụ nữ để lấy cảm hứng'),
    134: ('余小蓮 / 當當', 'Bất ngờ khi bị trượt.'),
    135: ('余小蓮 / 當當', 'Thất vọng vì không được chọn.'),
    138: ('龍傳人', 'Nhìn ngắm và khen ngợi.'),
    145: ('余小蓮 / 當當', 'Lúng túng chuẩn bị.'),
    146: ('龍傳人', 'Giục giã cấp dưới.'),
    147: ('龍傳人', 'Cười đắc ý khen ngợi.'),
    149: ('余小蓮 / 當當', 'Xấu hổ thú nhận chỉ thấy một chút.'),
    154: ('老伯', 'Tiếc nuối vì cảm xúc vừa dâng lên đã vụt'),
    155: ('胖公安', 'Than thở vì trạng thái hưng phấn vừa qua'),
    156: ('朱正軍 / 阿軍', 'Bực bội vì mất hứng.'),
}

#: 14 dòng có tên người KHÔNG xuất hiện trong nhãn ±10 dòng quanh đó.
UNSUPPORTED = [71, 72, 80, 83, 103, 105, 108, 111, 113, 114, 131, 132, 154, 156]


def _original() -> list[VisualCue]:
    return [VisualCue(line, spk, to, "") for line, (spk, to) in ORIGINAL.items()]


def _completed() -> list[VisualCue]:
    before = {cue.line_no: cue for cue in _original()}
    cues = dict(before)
    for line, (speaker, scene) in FILLS.items():
        addressee = before[line].addressee if line in before else ""
        cues[line] = VisualCue(line, speaker, addressee, scene)
    return sorted(cues.values(), key=lambda cue: cue.line_no)


def test_ha_dung_cac_ten_khong_co_mat_trong_canh() -> None:
    cues, findings = demote_unsupported_fills(
        _original(), _completed(), sorted(FILLS), ROSTER
    )
    assert [finding.line_no for finding in findings] == UNSUPPORTED
    assert {finding.reason for finding in findings} == {REASON_ABSENT}
    by_line = {cue.line_no: cue for cue in cues}
    for line in UNSUPPORTED:
        assert by_line[line].speaker == UNKNOWN_MALE_SPEAKER  # cả ba tên đều là nam
        assert by_line[line].scene == FILLS[line][1]  # mô tả giữ nguyên
    assert by_line[103].addressee == "龍傳人"  # người nghe của phân tích giữ nguyên


def test_giu_ten_co_mat_trong_canh() -> None:
    cues, _findings = demote_unsupported_fills(
        _original(), _completed(), sorted(FILLS), ROSTER
    )
    by_line = {cue.line_no: cue for cue in cues}
    # 老伯 vừa nói ở 100-101; 胖公安 được gọi ở 150-152; 龍傳人 nói khắp cảnh.
    assert by_line[104].speaker == "老伯"
    assert by_line[155].speaker == "胖公安"
    # Dòng 138: 14 lượt phân tích khác cũng nói là 龍傳人 → không hạ.
    assert by_line[138].speaker == "龍傳人"
    assert by_line[110].speaker == "Đám đông"  # nhãn dự phòng không khẳng định danh tính


def test_nhan_cu_khong_bi_dung() -> None:
    cues, _findings = demote_unsupported_fills(
        _original(), _completed(), sorted(FILLS), ROSTER
    )
    by_line = {cue.line_no: cue for cue in cues}
    for line, (speaker, addressee) in ORIGINAL.items():
        if speaker:
            assert by_line[line].speaker == speaker
            assert by_line[line].addressee == addressee


def test_canh_it_nhan_thi_khong_du_chung_cu() -> None:
    sparse = [VisualCue(100, "老伯", "龍傳人", ""), VisualCue(101, "老伯", "龍傳人", "")]
    completed = [*sparse, VisualCue(105, "朱正軍 / 阿軍", "", "Bày tỏ sự khó khăn")]
    cues, findings = demote_unsupported_fills(sparse, completed, [105], ROSTER)
    assert findings == []
    assert cues == completed


def test_khong_ro_gioi_tinh_thi_tra_ve_nhan_cu() -> None:
    roster = ROSTER.replace("朱正軍 / 阿軍 (Chu Chính Quân / A Quân) [nam, ~30]",
                            "朱正軍 / 阿軍 (Chu Chính Quân / A Quân)")
    cues, findings = demote_unsupported_fills(
        _original(), _completed(), sorted(FILLS), roster
    )
    by_line = {cue.line_no: cue for cue in cues}
    assert 105 in [finding.line_no for finding in findings]
    assert by_line[105] == VisualCue(105, "", "龍傳人", "")  # đúng nhãn trước khi hỏi
    assert 72 not in by_line  # dòng chưa từng có nhãn → bỏ hẳn


def _lines() -> list[TranslationLine]:
    lines = sorted(set(ORIGINAL) | set(FILLS))
    return [
        TranslationLine(index=line, start_ms=line * 2000, end_ms=line * 2000 + 1500,
                        text=f"câu {line}")
        for line in lines
    ]


def test_buoc_hoi_lai_dung_chung_luoi() -> None:
    answers = {line: speaker for line, (speaker, _scene) in FILLS.items()}

    def scan(lines: list[TranslationLine], _known: list[VisualCue]) -> list[VisualCue]:
        return [
            VisualCue(line.index, answers[line.index], "", FILLS[line.index][1])
            for line in lines if line.index in answers
        ]

    outcome = complete_and_check_speakers(_original(), _lines(), ROSTER, scan)
    assert list(outcome.demoted) == UNSUPPORTED
    assert outcome.after.missing == ()
    assert outcome.after.voice_only == len(UNSUPPORTED)


def test_phien_da_luu_nhan_ra_nhan_hoi_lai_nho_mo_ta() -> None:
    # Nhãn phân tích không có mô tả; nhãn hỏi lại luôn có — đúng như phiên thật (39/39).
    stored, findings = demote_unsupported_scan_labels(_completed(), ROSTER)
    assert [finding.line_no for finding in findings] == UNSUPPORTED
    assert sum(1 for cue in stored if cue.speaker == UNKNOWN_MALE_SPEAKER) == 14


def test_phien_khong_phan_biet_duoc_nguon_goc_thi_giu_nguyen() -> None:
    # Mọi nhãn đều có mô tả (phân tích ghi mô tả) → không biết nhãn nào là hỏi lại.
    described = [
        VisualCue(cue.line_no, cue.speaker, cue.addressee, cue.scene or "mô tả")
        for cue in _completed()
    ]
    stored, findings = demote_unsupported_scan_labels(described, ROSTER)
    assert findings == []
    assert stored == described


def test_long_tieng_khong_giao_vai_bia() -> None:
    from subtitles_extractor.application.services.dubbing_cast import (
        DubbingSource,
        line_speakers,
    )
    from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
    from subtitles_extractor.domain.value_objects.time_interval import TimeInterval

    # Cues đánh số theo dòng của CẢ phim: phụ đề TTS là đủ 165 dòng đầu.
    spans = tuple((line * 2.0, line * 2.0 + 1.5) for line in range(1, 166))
    source = DubbingSource(ROSTER, serialize_visual_cues(_completed()), spans)
    events = [
        SubtitleEvent(index=line, text=f"câu {line}", interval=TimeInterval(*span))
        for line, span in enumerate(spans, 1)
    ]
    speakers = dict(enumerate(line_speakers(events, source), 1))
    assert [speakers[line] for line in (103, 105, 154)] == [UNKNOWN_MALE_SPEAKER] * 3
    assert speakers[104] == "老伯"


def test_khau_dich_ha_nhan_da_luu_va_ghi_nhat_ky() -> None:
    from subtitles_extractor.application.services.translation_process_log import (
        get_process_log,
    )
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
    )

    request = TranslateSubtitlesRequest(
        events=[], stages=[],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER,
            visual_cues=serialize_visual_cues(_completed()),
        ),
    )
    log = get_process_log()
    mark = log.count()
    cleaned = TranslateSubtitlesUseCase._drop_contradicted_scan_cues(request, _lines())
    by_line = {
        cue.line_no: cue
        for cue in deserialize_visual_cues(cleaned.context.visual_cues)
    }
    kept_names = [
        line for line in UNSUPPORTED if by_line[line].speaker != UNKNOWN_MALE_SPEAKER
    ]
    assert kept_names == []
    events = [
        event for event in log.events_since(mark) if event["event"] == "scan_cues_demoted"
    ]
    assert events and events[0]["lines"] == UNSUPPORTED


# ── Model hỏi lại người nói ở khâu dịch ─────────────────────────────────────


class _Translator:
    """Bộ dịch giả: model trong ``failing`` luôn lỗi; ghi lại model đã được hỏi."""

    def __init__(self, failing: set[str], answers: dict[int, str]) -> None:
        self.failing = failing
        self.answers = answers
        self.models: list[str] = []

    def analyze_visual_cues(self, lines, target_lang, **kwargs):
        self.models.append(kwargs["model_name"])
        if kwargs["model_name"] in self.failing:
            raise RuntimeError("429 RESOURCE_EXHAUSTED (hết hạn mức ngày)")
        return [
            VisualCue(line.index, self.answers[line.index], "", "")
            for line in lines if line.index in self.answers
        ]


def _complete_in_translation(translator: _Translator, **request_fields):
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationContext,
    )

    request = TranslateSubtitlesRequest(
        events=[], stages=[],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER,
            visual_cues=serialize_visual_cues(_original()),
        ),
        enable_visual_cues=True, video_refs=("chunk",),
        visual_cues_model="gemini-3.5-flash-lite",
        **request_fields,
    )
    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    use_case._translator = translator
    return use_case._complete_speaker_cues(_lines(), request, None, None)


def test_hoi_lai_bang_model_phan_tich() -> None:
    translator = _Translator(set(), {line: "Đám đông" for line in FILLS})
    _complete_in_translation(translator, speaker_model="gemini-3.8-flash")
    assert translator.models == ["gemini-3.8-flash"]


def test_model_phan_tich_het_han_muc_thi_lui_ve_model_dich() -> None:
    translator = _Translator({"gemini-3.8-flash"}, {line: "Đám đông" for line in FILLS})
    request = _complete_in_translation(translator, speaker_model="gemini-3.8-flash")
    assert translator.models == ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    filled = {
        cue.line_no for cue in deserialize_visual_cues(request.context.visual_cues)
        if cue.speaker
    }
    assert set(FILLS) <= filled


def test_khong_chon_model_phan_tich_thi_dung_model_dich() -> None:
    translator = _Translator(set(), {line: "Đám đông" for line in FILLS})
    _complete_in_translation(translator)
    assert translator.models == ["gemini-3.5-flash-lite"]


def test_ca_hai_model_deu_loi_thi_giu_nhan_cu() -> None:
    translator = _Translator(
        {"gemini-3.8-flash", "gemini-3.5-flash-lite"}, {}
    )
    request = _complete_in_translation(translator, speaker_model="gemini-3.8-flash")
    assert json.loads(request.context.visual_cues) == json.loads(
        serialize_visual_cues(_original())
    )


def test_nguoi_dung_huy_thi_dung_ngay() -> None:
    from subtitles_extractor.application.use_cases.translate_subtitles import (
        TranslateSubtitlesRequest,
        TranslateSubtitlesUseCase,
    )
    from subtitles_extractor.domain.ports.subtitle_translator_port import (
        TranslationCancelledError,
        TranslationContext,
    )

    class _Cancelling(_Translator):
        def analyze_visual_cues(self, lines, target_lang, **kwargs):
            self.models.append(kwargs["model_name"])
            raise TranslationCancelledError("huỷ")

    translator = _Cancelling(set(), {})
    request = TranslateSubtitlesRequest(
        events=[], stages=[],
        context=TranslationContext(
            target_lang="Vietnamese", characters=ROSTER,
            visual_cues=serialize_visual_cues(_original()),
        ),
        enable_visual_cues=True, video_refs=("chunk",),
        speaker_model="gemini-3.8-flash",
    )
    use_case = TranslateSubtitlesUseCase.__new__(TranslateSubtitlesUseCase)
    use_case._translator = translator
    with pytest.raises(TranslationCancelledError):
        use_case._complete_speaker_cues(_lines(), request, None, lambda: True)
    assert translator.models == ["gemini-3.8-flash"]


def test_bo_quet_dan_khong_chia_ten_cho_dam_dong() -> None:
    from subtitles_extractor.infrastructure.translation import (
        gemini_translation_adapter as adapter,
    )

    instruction = adapter._visual_cues_system_instruction("Vietnamese", ROSTER)
    assert "KHÔNG chia lần lượt" in instruction
    assert "CÙNG một người" in instruction
