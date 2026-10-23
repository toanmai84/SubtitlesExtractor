"""[v3.23.599] Nạp vai lồng tiếng: đủ người nói cho MỌI dòng chưa, ĐÚNG người chưa.

Yêu cầu người dùng 10/10/2026: kiểm tra khi nạp vai từ trang Dịch thì các vai có được phân
đúng 100% số dòng và đúng người nói chưa. Đo trên 58 lượt dịch thật của 3 phim:

* Độ phủ: lượt mới nhất phim "灭门惨案II" đủ 1245/1245 dòng, nhưng phim "傻子落水…" còn
  103/3902 dòng và phim "he thong sung the" 125/5540 dòng không có người nói (đọc giọng
  chung) — trước bản này ứng dụng không báo.
* Đúng người: 85 dòng mà câu gốc GỌI TÊN chính người được gán là người nói rồi nói tiếp
  (dòng 495 "阿秀，你幫我" mang nhãn 秦可秀) — cả 85 đều khác nhãn đa số các lượt.
* Một nhân vật thành HAI vai: nhãn lai "王 Đại Phúc" (12 dòng) không khớp 王大福 trong
  bảng nhân vật → giọng khác chính nhân vật đó.

Dữ liệu: câu gốc, nhãn Visual Cues, dòng bảng nhân vật nguyên văn các lượt thật ghi trong
từng ca.
"""  # noqa: RUF002 — dấu phẩy chữ Hán trích nguyên văn

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonical_label,
    roster_label_map,
)
from subtitles_extractor.application.services.dubbing_cast import (
    DubbingSource,
    build_cast,
    line_speakers,
)
from subtitles_extractor.application.services.speaker_review import (
    REVIEW_GENDER,
    REVIEW_SELF_CALL,
    REVIEW_UNKNOWN,
    SpeakerIssue,
    leading_call,
    nickname_index,
    review_speakers,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
)
from tests.unit.presentation.test_dubbing_bugfixes_v323594 import _coordinator
from tests.unit.presentation.test_dubbing_line_edits_v323598 import (
    HUSBAND,
    LINES,
    P495,
    ROSTER,
    WIFE,
    _events,
)

#: Dòng bảng nhân vật thật, lượt ``44887efa`` (phim "灭门惨案II", 10/10/2026).
MM_HONG_WIFE = (
    "洪震之妻 / 洪大嫂 (Vợ Hồng Chấn / Hồng Đại Tẩu) [nữ, ~35] — Vợ của Hồng Chấn, mẹ "
    "của Hồng Chí Quyền."
)
MM_HONG = "洪震 (Hồng Chấn) [nam, ~40] — Kẻ lưu manh khét tiếng làng Đông Long."
MM_LONG = (
    "龍傳人 (Long Truyền Nhân) [nam, ~35] — Nhân vật chính, Trưởng đồn công an làng "
    "Đông Long."
)
MM_AIQUOC = (
    "愛國 (Ái Quốc) [nam, ~25] — Công an viên đồn Đông Long: Cấp dưới thích nịnh bợ Long "
    "Truyền Nhân. Cách xưng hô: với 龍傳人 tự xưng 'em' gọi 'Trưởng đồn'."
)
#: Dòng bảng nhân vật thật, lượt ``da77ec5a`` (phim "傻子落水…").
SZ_MINGYUE = (
    "沈明月 (Thẩm Minh Nguyệt) [nữ, ~26] — Chủ tịch Khách sạn Đế Hào, bạn thân của Liễu "
    "Mộ Vân."
)
SZ_MAZI = (
    "王麻子 (Vương Ma Tử) [nam, ~35] — tên lưu manh mặt rỗ trong thôn, cậy tiền sính lễ "
    "ép cưới Đào Hoa."
)
#: Dòng bảng nhân vật thật, lượt ``069d0c24`` (phim "he thong sung the").
HT_DAFU = (
    "王大福 / 胖子 / 三胖子 (Vương Đại Phúc / Bàn Tử / Tam Bàn Tử) [nam, ~24] — bạn thân "
    "nối khố của Lâm Dật."
)
HT_MAI = (
    "王梅 (Vương Mai) [nữ, ~32] — bà chủ tiệm tạp hóa: quyến rũ, nóng bỏng và thích "
    "trêu ghẹo Lâm Dật. Xưng hô: với Lâm Dật tự xưng 'chị' gọi 'nhóc'"
)
HT_QINYUE = "秦月 (Tần Nguyệt) [nữ, ~24] — bạn gái cũ của Lâm Dật."
HT_LINYI = "林逸 (Lâm Dật) [nam, ~24] — nhân vật chính."
DAFU = "王大福 / 胖子 / 三胖子"
#: Lượt ``069d0c24`` dòng 4370-4378: (dòng, start, end, người nói, người nghe, câu gốc).
HT_LINES = [
    (4370, 7405.0, 7406.067, "林逸", "王 Đại Phúc", "区区两百万"),
    (4371, 7406.133, 7406.8, "林逸", "王 Đại Phúc", "就让这女人"),
    (4372, 7406.867, 7407.6, "林逸", "王 Đại Phúc", "露出原型了"),
    (4373, 7407.667, 7408.733, "王 Đại Phúc", "林逸", "不过就这么点钱"),
    (4374, 7408.8, 7410.267, "王 Đại Phúc", "林逸", "能让她答应捐骨髓吗"),
    (4375, 7410.333, 7411.467, "林逸", DAFU, "这点钱啊"),
]


def _review(
    lines: list[tuple[str, str]], characters: str, skip: frozenset[int] = frozenset()
) -> dict[int, SpeakerIssue]:
    """``lines`` = [(người nói, câu gốc)]."""
    speakers = [speaker for speaker, _text in lines]
    genders = {member.name: member.gender for member in build_cast(speakers, characters)}
    return review_speakers(
        speakers, [text for _speaker, text in lines], characters, genders, skip
    )


# ── Một nhân vật, một vai ────────────────────────────────────────────────────


class TestMixedScriptLabels:
    def test_mixed_labels_reach_the_roster_name(self) -> None:
        label_map = roster_label_map("\n".join([HT_DAFU, HT_MAI, HT_LINYI]))
        assert canonical_label("王 Đại Phúc", label_map) == DAFU
        assert canonical_label("王Đại Phúc", label_map) == DAFU
        assert canonical_label("王 Mai", label_map) == "王梅"
        assert canonical_label("Vương Đại Phúc", label_map) == DAFU

    def test_other_readings_are_left_alone(self) -> None:
        """"洪 Đại Sảo" (lượt ``d142948e``) ≠ phiên âm "Hồng Đại Tẩu" của bảng mới."""
        label_map = roster_label_map(MM_HONG_WIFE)
        assert canonical_label("洪 Đại Sảo", label_map) == "洪 Đại Sảo"

    def test_one_character_is_one_dubbing_role(self) -> None:
        roster = "\n".join([HT_DAFU, HT_MAI, HT_LINYI])
        cues = [
            {"id": i, "spk": row[3], "to": row[4], "cue": ""}
            for i, row in enumerate(HT_LINES, 1)
        ]
        source = DubbingSource(
            roster, json.dumps(cues, ensure_ascii=False),
            tuple((row[1], row[2]) for row in HT_LINES),
        )
        events = [
            SubtitleEvent(index=i, text=row[5], interval=TimeInterval(row[1], row[2]))
            for i, row in enumerate(HT_LINES, 1)
        ]
        speakers = line_speakers(events, source)
        assert speakers[3:5] == [DAFU, DAFU]
        cast = {member.name: member.gender for member in build_cast(speakers, roster)}
        assert cast == {"林逸": VOICE_MALE, DAFU: VOICE_MALE}


# ── Câu gốc cho thấy nhãn sai ────────────────────────────────────────────────


class TestSelfCall:
    def test_nicknames(self) -> None:
        index = nickname_index(ROSTER)
        assert index["阿秀"] == WIFE
        assert index["可秀"] == WIFE
        assert index["劍聲"] == HUSBAND
        assert "小心" not in nickname_index("林小心 (Lâm Tiểu Tâm) [nữ] — x")

    def test_real_wrong_labels_are_flagged(self) -> None:
        roster = "\n".join([ROSTER, MM_HONG_WIFE, MM_HONG, SZ_MINGYUE])
        hong_wife = "洪震之妻 / 洪大嫂"
        issues = _review([
            (WIFE, LINES[P495][6]),                 # 44887efa dòng 495
            (hong_wife, "洪大嫂我不想吃"),          # 44887efa dòng 841
            ("沈明月", "明月姐，你别闹了"),  # noqa: RUF001 — ea37e87a dòng 2992
            ("沈明月", "沈明月你放开"),             # ea37e87a dòng 3291
        ], roster)
        assert issues == {
            0: SpeakerIssue(REVIEW_SELF_CALL, WIFE, "阿秀"),
            1: SpeakerIssue(REVIEW_SELF_CALL, hong_wife, "洪大嫂"),
            2: SpeakerIssue(REVIEW_SELF_CALL, "沈明月", "明月"),
            3: SpeakerIssue(REVIEW_SELF_CALL, "沈明月", "沈明月"),
        }

    def test_calling_someone_else_or_only_a_name_is_fine(self) -> None:
        roster = "\n".join([ROSTER, HT_QINYUE])
        issues = _review([
            (WIFE, LINES[4][6]),  # dòng 494 "劍哥…": vợ gọi chồng — đúng
            ("秦月", "秦月"),           # 069d0c24 dòng 5189: 7/7 lượt giữ nhãn này
            (HUSBAND, LINES[P495][6]),  # 495 sau khi sửa đúng người
        ], roster)
        assert issues == {}

    def test_leading_call_needs_more_words(self) -> None:
        index = nickname_index(ROSTER)
        assert leading_call("阿秀，你幫我，幫幫我..", index) == (WIFE, "阿秀")  # noqa: RUF001
        assert leading_call("阿秀……", index) is None
        assert leading_call("「阿秀，你幫我", index) == (WIFE, "阿秀")  # noqa: RUF001


class TestGenderAndUnknown:
    def test_self_reference_against_role_gender(self) -> None:
        """Lượt ``7ed55a62`` dòng 1241: "你们让我嫁给他" mang nhãn 王麻子 (nam)."""
        issues = _review([("王麻子", "你们让我嫁给他")], SZ_MAZI)
        assert issues == {0: SpeakerIssue(REVIEW_GENDER, VOICE_FEMALE, "我嫁")}

    def test_unknown_and_hand_picked_lines(self) -> None:
        issues = _review(
            [("", "啊"), ("", "发生什么了?"), (WIFE, LINES[P495][6])],
            ROSTER, skip=frozenset({1, 2}),
        )
        assert issues == {0: SpeakerIssue(REVIEW_UNKNOWN)}


# ── Bộ điều phối + trang TTS ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "tts.ini"), QSettings.Format.IniFormat)


def _source() -> DubbingSource:
    cues = [
        {"id": i, "spk": row[4], "to": row[5], "cue": ""}
        for i, row in enumerate(LINES, 1)
    ]
    return DubbingSource(
        characters=ROSTER,
        visual_cues=json.dumps(cues, ensure_ascii=False),
        source_spans=tuple((row[1], row[2]) for row in LINES),
        source_texts=tuple(row[6] for row in LINES),
    )


class TestCoordinator:
    def test_wrong_line_is_marked_until_fixed(
        self, app: QApplication, settings: QSettings
    ) -> None:
        coordinator, panel, view_model = _coordinator(settings, _events())
        coordinator.set_source_provider(_source)
        panel.set_dubbing_enabled(True)
        assert coordinator.load_cast()
        assert coordinator.review_positions() == [P495]
        cell = coordinator.line_speaker_cells()[P495]
        assert cell.needs_review
        assert cell.text == "⚠ Tần Khả Tú · HoaiMy (nữ)"
        assert '"阿秀"' in cell.tooltip and "Tần Khả Tú" in cell.tooltip
        assert "⚠ 1 dòng nhãn người nói chắc sai." in panel.status()

        coordinator.set_line_speaker([P495], HUSBAND)
        assert coordinator.review_positions() == []
        assert "chắc sai" not in panel.status()
        assert view_model.speaker_issues() == {}

    def test_lines_of_another_film_are_unknown(
        self, app: QApplication, settings: QSettings
    ) -> None:
        from tests.unit.presentation.test_dubbing_bugfixes_v323594 import (
            OTHER_FILM,
            _other_events,
        )

        events = [*_other_events(), replace(
            _other_events()[0], text=f"[{HUSBAND}:] Chào em.",
        )]
        coordinator, panel, _vm = _coordinator(settings, events)
        coordinator.set_source_provider(_source)
        panel.set_dubbing_enabled(True)
        assert coordinator.load_cast()  # chỉ dòng có tag
        assert coordinator.review_positions() == list(range(len(OTHER_FILM)))
        assert f"⚠ {len(OTHER_FILM)} dòng chưa rõ người nói" in panel.status()

    def test_roles_of_unknown_gender_are_listed(
        self, app: QApplication, settings: QSettings
    ) -> None:
        """Lượt ``44887efa`` dòng 186-205: bảng nhân vật mới nhất bỏ mất 阿積 (18/19 lượt
        trước có), 6 dòng nhãn 阿積 thành vai chưa rõ giới tính."""
        rows = [
            (186, 787.467, 788.667, "龍傳人", "隨傳隨到"),
            (187, 791.467, 793.8, "龍傳人", "阿積有結果沒有"),
            (188, 794.467, 797.0, "阿積", "這麼多，驗尿都沒有那麼快啦"),  # noqa: RUF001
            (189, 797.067, 798.133, "阿積", "又不是這一支"),
            (190, 798.8, 800.4, "龍傳人", "這種事不能急呀"),
            (196, 816.0, 816.8, "愛國", "不好意思"),
            (201, 841.8, 842.8, "阿積", "所長，有發現呀"),  # noqa: RUF001
            (202, 844.733, 845.733, "阿積", "這瓶精液"),
        ]
        cues = [
            {"id": i, "spk": row[3], "to": "", "cue": ""} for i, row in enumerate(rows, 1)
        ]

        def source() -> DubbingSource:
            return DubbingSource(
                "\n".join([MM_LONG, MM_AIQUOC]), json.dumps(cues, ensure_ascii=False),
                tuple((row[1], row[2]) for row in rows), tuple(row[4] for row in rows),
            )

        events = [
            SubtitleEvent(index=i, text=row[4], interval=TimeInterval(row[1], row[2]))
            for i, row in enumerate(rows, 1)
        ]
        coordinator, panel, _vm = _coordinator(settings, events)
        coordinator.set_source_provider(source)
        panel.set_dubbing_enabled(True)
        assert coordinator.load_cast()
        assert [m.name for m in coordinator.unknown_gender_roles()] == ["阿積"]
        assert "1 vai chưa rõ giới tính (4 dòng: 阿積)" in panel.status()


@pytest.fixture
def page(app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from subtitles_extractor.composition.container import ApplicationContainer
    from subtitles_extractor.infrastructure.settings.json_file_repository import (
        JsonFileRepository,
    )
    from subtitles_extractor.presentation.pages import tts_page as tts_page_module

    monkeypatch.setattr(
        tts_page_module, "QSettings",
        lambda *_args: QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat),
    )
    container = ApplicationContainer(
        settings_repository=JsonFileRepository(file_path=tmp_path / "config.json"),
        i18n_data_dir=Path(tts_page_module.__file__).resolve().parents[2] / "data",
        user_data_dir=tmp_path,
    )
    tts = tts_page_module.TTSPage(container)
    tts.set_dubbing_source_provider(_source)
    return tts


def test_review_button_selects_the_wrong_line(page) -> None:
    page._dubbing_panel.set_dubbing_enabled(True)
    page._view_model.set_source_events(_events())
    controller = page._line_edits
    assert controller._btn_review.text().endswith("(1)")
    assert controller._btn_review.isEnabled()
    assert controller.select_review() == 1
    assert page._selected_event_indices() == [P495]
    assert page._table.item(P495, 7).text().startswith("⚠ Tần Khả Tú")
    controller.apply_speaker([P495], HUSBAND)
    controller.refresh()
    assert not controller._btn_review.isEnabled()
