"""[v3.23.593] Phân vai lồng tiếng: người nói từng dòng, danh sách vai, giao giọng.

Dữ liệu chép nguyên văn từ lượt dịch thật v3.23.592 (phim "灭门惨案II", ``0c0c7c9a``):
dòng bảng nhân vật, nhãn người nói/người nghe của Visual Cues, mốc thời gian câu gốc và
bản dịch cuối (dòng 2: bản dịch kéo dài mốc kết thúc 110.6 → 110.99). Cue ``id`` được
đánh lại theo vị trí trong đoạn trích (1..n) — đúng quy ước của cue thật (vị trí câu
gốc, từ 1).

Trên cả phim (1246 dòng): 1245 dòng biết người nói, 540 lần đổi người nói giữa hai dòng
liền nhau. Chỉ hai giọng Edge → 251/540 lần hai người đối đáp cùng một giọng; thêm biến
thể trầm/cao (6 giọng) → 8/540; 30 giọng Gemini → 0.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from subtitles_extractor.application.services.dubbing_cast import (
    AGE_ADULT,
    AGE_CHILD,
    AGE_ELDER,
    DubbingSource,
    auto_cast,
    build_cast,
    conversation_weights,
    line_speakers,
    voices_for_lines,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    TONE_HIGH,
    TONE_LOW,
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
)

ROSTER = "\n".join([
    "龍傳人 (Long Truyền Nhân) [nam, ~30] — nhân vật chính: Sở trưởng đồn công an "
    "Đông Long hám danh nhưng giàu lòng trắc ẩn. Trong đoạn này, anh tìm gặp Tần "
    "Khả Tú tại mộ chồng cô và thuyết phục cô về đồn công an để hợp tác điều "
    "tra. Cách xưng hô: với Tần Khả Tú tự xưng 'tôi' gọi 'chị' (danh xưng: Chị "
    "dâu Chân); với A Tích tự xưng 'tôi' gọi 'cậu'.",
    "阿積 (A Tích) [nam, ~25] — công an viên đồn Đông Long, tính nịnh bợ. Xuất "
    "hiện ngắn ngủi ở đầu đoạn khi chuẩn bị đi nhậu ăn mừng cùng Long Truyền "
    "Nhân. Cách xưng hô: tự xưng 'tôi' gọi Long Truyền Nhân là 'Trưởng đồn'.",
    "余小蓮 (Dư Tiểu Liên) [nữ, ~23] — Nữ công an viên đồn Đông Long, bị Long "
    "Truyền Nhân ép cởi áo sau rèm để làm mồi nhử kích dục cho dân làng lấy mẫu "
    "xét nghiệm. Xưng hô: với Long Truyền Nhân tự xưng 'tôi' gọi 'Trưởng đồn'.",
    "老莫 (Lão Mạc) [nam, trung niên] — Dân làng Đông Long: tát Long Truyền Nhân "
    "vì tội sàm sỡ người vợ đang ngất xỉu của mình. Cách xưng hô: xưng 'tao' gọi "
    "'mày' khi tức giận.",
    "Cậu bé trong thôn [nam, ~10] — cậu bé thông minh trong thôn, tự tin đứng ra "
    "giải thích cặn kẽ tiếng lóng dâm dục cho người dân hiểu.",
    "Lão nhân trong thôn [nam, ~70] — cụ già trong thôn, mang theo chiếc xô to "
    "để nộp tinh dịch và lo lắng không biết có đủ định lượng hay không.",
])
#: (dòng thật, start gốc, end gốc, end bản dịch, người nói, người nghe, bản dịch cuối)
ROWS = [
    (1, 107.133, 109.067, 109.067, "龍傳人", "龍傳人", "Phá một vụ thảm án diệt môn,"),
    (
        2, 109.133, 110.6, 110.99, "龍傳人", "龍傳人",
        "thì lập tức thăng chức Cục trưởng ngay.",
    ),
    (
        3, 111.467, 115.933, 115.933, "龍傳人", "龍傳人",
        "Đúng là quan vận hanh thông, tinh thần phơi phới.",
    ),
    (43, 254.267, 255.067, 255.067, "老莫", "龍傳人", "bị diệt môn cả nhà,"),
    (44, 255.333, 256.6, 256.6, "老莫", "龍傳人", "mày lợi dụng lúc vợ tao ngất xỉu"),
    (45, 256.8, 257.4, 257.4, "老莫", "龍傳人", "để sàm sỡ cô ấy."),
    (46, 257.467, 258.4, 258.4, "老莫", "龍傳人", "Mày đúng là đồ khốn!"),
    (47, 258.533, 259.0, 259.0, "龍傳人", "老莫", "Tôi..."),
    (48, 259.133, 260.733, 260.733, "余小蓮", "龍傳人", "Trưởng đồn."),
    (49, 260.8, 260.9, 260.9, "余小蓮", "龍傳人", "Trưởng đồn..."),
    (50, 261.0, 262.867, 262.867, "龍傳人", "余小蓮", "Lại có nhà chết sạch nữa à?"),
    (51, 263.067, 264.333, 264.333, "老莫", "龍傳人", "Mày đúng là khốn nạn!"),
    (52, 297.4, 298.467, 298.467, "余小蓮", "龍傳人", "Trưởng đồn ạ."),
    (53, 300.667, 302.333, 302.333, "余小蓮", "龍傳人", "Trưởng đồn, còn một vụ nữa."),
    (54, 307.6, 308.667, 308.667, "龍傳人", "余小蓮", "Để tôi, cô ra ngoài đi."),
    (
        55, 320.267, 322.133, 322.133, "龍傳人", "余小蓮",
        "Quả là một vụ diệt môn kinh hoàng.",
    ),
    (56, 334.933, 335.8, 335.8, "阿積", "龍傳人", "Trưởng đồn,"),
    (
        85, 451.6, 453.733, 453.733, "Cậu bé trong thôn", "Người dân trong thôn",
        "Tôi biết!",
    ),
    (
        100, 480.467, 485.4, 485.4, "Lão nhân trong thôn", "龍傳人",
        "Đội trưởng, tôi làm được, nhưng có cần phải chứa đầy không?",
    ),
]
EDGE_VI = [
    VoiceProfile("vi-VN-HoaiMyNeural", gender=VOICE_FEMALE),
    VoiceProfile("vi-VN-HoaiMyNeural@-12Hz", gender=VOICE_FEMALE, tone=TONE_LOW),
    VoiceProfile("vi-VN-HoaiMyNeural@+12Hz", gender=VOICE_FEMALE, tone=TONE_HIGH),
    VoiceProfile("vi-VN-NamMinhNeural", gender=VOICE_MALE),
    VoiceProfile("vi-VN-NamMinhNeural@-12Hz", gender=VOICE_MALE, tone=TONE_LOW),
    VoiceProfile("vi-VN-NamMinhNeural@+12Hz", gender=VOICE_MALE, tone=TONE_HIGH),
]


def _source(rows: list[tuple] = ROWS) -> DubbingSource:
    cues = [
        {"id": position, "spk": speaker, "to": listener, "cue": ""}
        for position, (_line, _s, _e, _fe, speaker, listener, _text) in enumerate(rows, 1)
    ]
    return DubbingSource(
        characters=ROSTER,
        visual_cues=json.dumps(cues, ensure_ascii=False),
        source_spans=tuple((start, end) for _l, start, end, *_rest in rows),
    )


def _events(rows: list[tuple] = ROWS) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(start_sec=start, end_sec=final_end, text=text)
        for _line, start, _end, final_end, _spk, _to, text in rows
    ]


def _speakers() -> list[str]:
    return line_speakers(_events(), _source())


class TestLineSpeakers:
    def test_real_case_every_line_gets_its_cue_speaker(self) -> None:
        assert _speakers() == [row[4] for row in ROWS]

    def test_translation_with_extended_end_still_maps_to_its_own_line(self) -> None:
        """Dòng 2: bản dịch kéo mốc kết thúc 110.6 → 110.99 — vẫn là câu gốc dòng 2."""
        rows = ROWS[:3]
        source = _source(rows)
        shifted = DubbingSource(
            source.characters,
            json.dumps([
                {"id": 1, "spk": "龍傳人"}, {"id": 2, "spk": "老莫"},
                {"id": 3, "spk": "余小蓮"},
            ], ensure_ascii=False),
            source.source_spans,
        )
        assert line_speakers(_events(rows), shifted) == ["龍傳人", "老莫", "余小蓮"]

    def test_speaker_tag_in_the_line_wins_over_the_cue(self) -> None:
        """Tag ``[Tên:]`` là người dùng tự ghi — đúng hơn cue máy đoán."""
        events = _events()
        events[8] = SimpleNamespace(
            start_sec=events[8].start_sec, end_sec=events[8].end_sec,
            text="[Long Truyền Nhân:] Trưởng đồn.",
        )
        speakers = line_speakers(events, _source())
        assert speakers[8] == "龍傳人"

    def test_tags_alone_are_enough_without_visual_cues(self) -> None:
        events = [
            SimpleNamespace(
                start_sec=1.0, end_sec=2.0, text="[Dư Tiểu Liên:] Trưởng đồn."
            ),
            SimpleNamespace(start_sec=3.0, end_sec=4.0, text="Không có tag."),
        ]
        assert line_speakers(events, DubbingSource(characters=ROSTER)) == ["余小蓮", ""]

    def test_line_without_overlap_falls_back_to_position_when_counts_match(self) -> None:
        events = _events()
        events[0] = SimpleNamespace(start_sec=0.0, end_sec=0.0, text=events[0].text)
        assert line_speakers(events, _source())[0] == "龍傳人"

    def test_no_context_means_no_speaker(self) -> None:
        assert set(line_speakers(_events(), DubbingSource())) == {""}


class TestBuildCast:
    def test_real_roster_traits(self) -> None:
        cast = {member.name: member for member in build_cast(_speakers(), ROSTER)}
        assert cast["龍傳人"].display_name == "Long Truyền Nhân"
        lead = cast["龍傳人"]
        assert (lead.gender, lead.age_group) == (VOICE_MALE, AGE_ADULT)
        assert cast["余小蓮"].gender == VOICE_FEMALE
        assert cast["老莫"].age_group == AGE_ADULT  # "trung niên"
        assert cast["Cậu bé trong thôn"].age_group == AGE_CHILD  # "~10"
        assert cast["Lão nhân trong thôn"].age_group == AGE_ELDER  # "~70"

    def test_most_lines_first(self) -> None:
        names = [member.name for member in build_cast(_speakers(), ROSTER)]
        assert names[:3] == ["龍傳人", "老莫", "余小蓮"]
        assert [m.line_count for m in build_cast(_speakers(), ROSTER)][:3] == [7, 5, 4]

    def test_label_outside_the_roster_is_still_a_role(self) -> None:
        cast = build_cast(["Người dân trong thôn", "Bà lão bán hàng"], ROSTER)
        by_name = {member.name: member for member in cast}
        assert by_name["Người dân trong thôn"].in_roster is False
        assert by_name["Người dân trong thôn"].gender == ""
        assert by_name["Bà lão bán hàng"].gender == VOICE_FEMALE


class TestAutoCast:
    def _casting(self, catalog: list[VoiceProfile]) -> dict[str, str]:
        speakers = _speakers()
        return auto_cast(
            build_cast(speakers, ROSTER), catalog, conversation_weights(speakers)
        )

    def test_gender_is_respected(self) -> None:
        casting = self._casting(EDGE_VI)
        assert casting["余小蓮"].startswith("vi-VN-HoaiMy")
        men = ("龍傳人", "老莫", "阿積", "Cậu bé trong thôn", "Lão nhân trong thôn")
        for name in men:
            assert casting[name].startswith("vi-VN-NamMinh"), name

    def test_characters_who_talk_to_each_other_get_different_voices(self) -> None:
        speakers = _speakers()
        casting = self._casting(EDGE_VI)
        clashes = [
            pair for pair in conversation_weights(speakers)
            if len({casting[name] for name in pair}) == 1
        ]
        assert clashes == []

    def test_age_picks_the_tone(self) -> None:
        """Trẻ con lấy biến thể cao, cụ già lấy biến thể trầm khi còn trống."""
        casting = auto_cast(
            build_cast(["Cậu bé trong thôn", "Lão nhân trong thôn"], ROSTER), EDGE_VI, {}
        )
        assert casting["Cậu bé trong thôn"] == "vi-VN-NamMinhNeural@+12Hz"
        assert casting["Lão nhân trong thôn"] == "vi-VN-NamMinhNeural@-12Hz"

    def test_two_voices_only_still_casts_everyone(self) -> None:
        catalog = [EDGE_VI[0], EDGE_VI[3]]
        casting = self._casting(catalog)
        assert set(casting.values()) <= {"vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"}
        assert len(casting) == 6

    def test_unknown_gender_takes_any_voice(self) -> None:
        cast = build_cast(["Người dân trong thôn"], ROSTER)
        assert auto_cast(cast, [EDGE_VI[0]], {}) == {
            "Người dân trong thôn": "vi-VN-HoaiMyNeural"
        }

    def test_no_voices_no_casting(self) -> None:
        assert auto_cast(build_cast(_speakers(), ROSTER), [], {}) == {}


class TestVoicesForLines:
    def test_maps_each_line_and_leaves_unknown_lines_to_the_shared_voice(self) -> None:
        voices = voices_for_lines(["A", "", "B", "C"], {"A": "x", "B": "y"})
        assert voices == ("x", "", "y", "")

    def test_empty_when_nobody_has_a_voice(self) -> None:
        assert voices_for_lines(["A", ""], {}) == ()
