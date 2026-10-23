"""[v3.23.614] Bảng nhân vật có câu "Xưng hô: …" viết hoa trong mô tả vẫn phải đọc đủ.

Ca thật 14/10 (phim "天骄女友退婚", tập 1, gemini-3.8-flash, thinking cao): mô tả của mỗi
nhân vật kết thúc bằng một câu mới "… Luyện Khí. Xưng hô: với Sở Dao tự xưng 'huynh' gọi
'muội'." Bộ đọc coi "Xưng hô:" sau dấu chấm là tên một nhân vật THỨ HAI trên dòng nên bỏ
cách đọc tất định, rơi xuống nhánh dự phòng: bảng 7 người còn 4 tên trơn — mất nữ chính
楚瑶 và hai đệ tử, mất mọi giới tính, phiên âm và quy tắc xưng hô. Hệ quả dây chuyền: khâu
dịch báo "KHÔNG có quy tắc xưng hô nào", chẩn đoán ngữ cảnh báo 0 quy tắc, lồng tiếng đọc
cả 94 dòng bằng một giọng nam.
"""

from __future__ import annotations

from subtitles_extractor.application.services.roster_format import (
    address_rule_map,
    parse_roster,
)

#: Bảng nhân vật thật (nguyên văn) của lượt phân tích 14/10.
ROSTER = "\n".join([
    "叶然 (Diệp Nhiên) [nam, ~20] — Nhân vật chính, đệ tử Tiểu Vân Phong thuộc Linh "
    "Tiêu Tông; người xuyên không mang hệ thống, tu vi đã đạt Đại Thừa cảnh đại "
    "viên mãn nhưng hễ rời khỏi sơn môn là bị phong ấn tu vi về Luyện Khí. Xưng "
    "hô: với Sở Dao tự xưng 'huynh' gọi 'muội'.",
    "楚瑶 (Sở Dao) [nữ, ~19] — Sư muội kiêm vị hôn thê của Diệp Nhiên, thiên kiêu "
    "Linh Tiêu Tông, vừa bế quan đạt Trúc Cơ viên mãn, đến để từ hôn với Diệp "
    "Nhiên. Xưng hô: với Diệp Nhiên tự xưng 'muội' gọi 'sư huynh'.",
    "女弟子 (Nữ đệ tử) [nữ, ~18] — Đệ tử Linh Tiêu Tông đi cùng Sở Dao, nịnh bợ Sở "
    "Dao và châm biếm Diệp Nhiên. Xưng hô: với Diệp Nhiên gọi 'ngươi', với Sở Dao "
    "gọi 'Sở sư tỷ'.",
    "男弟子 (Nam đệ tử) [nam, ~20] — Đệ tử Linh Tiêu Tông đi theo Sở Dao, khinh "
    "thường và mỉa mai tu vi của Diệp Nhiên. Xưng hô: với Diệp Nhiên xưng 'ta' gọi "
    "'ngươi', với Sở Dao gọi 'sư tỷ'.",
    "萧炎 (Tiêu Viêm) [nam, thanh niên] — Nhân vật chính trong tiểu thuyết tu tiên "
    "Đấu Phá Thương Khung, được Diệp Nhiên nhắc đến trong suy nghĩ khi nhớ về kịch "
    "bản phế tài bị từ hôn.",
    "师尊 / 师父 (Sư tôn / Sư phụ) [nam, cao tuổi] — Sư phụ của Diệp Nhiên tại Tiểu "
    "Vân Phong thuộc Linh Tiêu Tông, hiện đang bế tử quan.",
    "两位长老 (Hai vị trưởng lão) [nam, cao tuổi] — Các trưởng lão của Linh Tiêu Tông "
    "từng đứng ra đính ước hôn sự môn đăng hộ đối cho Diệp Nhiên và Sở Dao.",
])


class TestSentenceLabelInRole:
    def test_every_character_is_read_with_details(self) -> None:
        entries = parse_roster(ROSTER)
        assert [entry.name for entry in entries] == [
            "叶然", "楚瑶", "女弟子", "男弟子", "萧炎", "师尊 / 师父", "两位长老",
        ]
        assert [entry.gender for entry in entries] == [
            "nam", "nữ", "nữ", "nam", "nam", "nam", "nam",
        ]
        assert entries[1].aka == "Sở Dao"
        assert entries[0].role.endswith("tự xưng 'huynh' gọi 'muội'.")

    def test_address_rules_are_read(self) -> None:
        rules = address_rule_map(ROSTER)
        assert ("叶然", "楚瑶") in rules
        assert "'muội'" in rules[("叶然", "楚瑶")]
        assert ("楚瑶", "叶然") in rules
        assert "'sư huynh'" in rules[("楚瑶", "叶然")]

    def test_legacy_packed_line_still_falls_back(self) -> None:
        """Dữ liệu cũ gói nhiều người trên một dòng vẫn phải tách đúng."""
        names = [e.name for e in parse_roster("Lâm Côn: nam chính. Tô Mai: nữ chính.")]
        assert names == ["Lâm Côn", "Tô Mai"]


class TestDiagnosticsSafetyNet:
    """Chẩn đoán ngữ cảnh phải BÁO khi bộ đọc bỏ sót dòng của bảng nhân vật.

    Lượt 14/10 xuất chẩn đoán với ``quality_flags: []`` trong khi bảng 7 người chỉ đọc
    được 4 tên trơn và 0 quy tắc xưng hô — lỗi chỉ lộ ra nhờ soi tay.
    """

    def test_unread_lines_are_listed(self) -> None:
        from subtitles_extractor.application.services.roster_format import (
            RosterEntry,
            unread_roster_lines,
        )

        partial = [RosterEntry(name="叶然"), RosterEntry(name="萧炎")]
        unread = unread_roster_lines(ROSTER, partial)
        assert [line.split(" ", 1)[0] for line in unread] == [
            "楚瑶", "女弟子", "男弟子", "师尊", "两位长老",
        ]
        assert unread_roster_lines(ROSTER, parse_roster(ROSTER)) == []

    def test_heading_line_is_not_reported(self) -> None:
        from subtitles_extractor.application.services.roster_format import (
            unread_roster_lines,
        )

        text = "Xưng hô\n叶然 (Diệp Nhiên) [nam, ~20] — nhân vật chính"
        assert unread_roster_lines(text, parse_roster(text)) == []

    def test_report_flags_unread_lines_and_missing_rules(
        self, monkeypatch: object
    ) -> None:
        from types import SimpleNamespace

        from subtitles_extractor.application.services import context_diagnostics
        from subtitles_extractor.application.services.roster_format import RosterEntry

        broken = [RosterEntry(name="叶然"), RosterEntry(name="萧炎")]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            context_diagnostics, "parse_roster", lambda _text: broken
        )
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="", characters=ROSTER,
            glossary="", visual_cues="",
        )
        report = context_diagnostics.build_context_diagnostics(context=context)
        codes = {flag["code"] for flag in report["quality_flags"]}
        assert "roster_lines_unread" in codes
        assert report["roster"]["unread_lines"]

    def test_roster_without_rules_is_flagged(self) -> None:
        from types import SimpleNamespace

        from subtitles_extractor.application.services import context_diagnostics

        roster = "\n".join(
            f"{name} ({reading}) [nam, ~30] — đệ tử" for name, reading in
            [("张三", "Trương Tam"), ("李四", "Lý Tứ"), ("王五", "Vương Ngũ")]
        )
        context = SimpleNamespace(
            target_lang="Vietnamese", source_lang="zh", overview="", characters=roster,
            glossary="", visual_cues="",
        )
        report = context_diagnostics.build_context_diagnostics(context=context)
        codes = {flag["code"] for flag in report["quality_flags"]}
        assert "roster_without_address_rules" in codes


def test_merging_segments_keeps_every_character() -> None:
    """Phân tích nhiều đoạn gộp bảng đoạn trước (văn bản) với bảng đoạn mới.

    Bản 3.23.613 đọc lại bảng đoạn trước bằng bộ đọc lỗi → từ đoạn thứ hai, bảng chỉ còn
    tên trơn của 4 người: nữ chính và mọi giới tính, phiên âm, xưng hô mất hẳn.
    """
    from subtitles_extractor.application.services.roster_format import merge_roster

    newcomer = "王长老 (Vương trưởng lão) [nam, ~70] — trưởng lão Linh Tiêu Tông"
    merged = parse_roster(merge_roster(ROSTER, newcomer))
    assert len(merged) == 8
    assert merged[1].name == "楚瑶" and merged[1].gender == "nữ"
    assert "tự xưng 'huynh' gọi 'muội'" in merged[0].role


def test_render_then_parse_is_lossless() -> None:
    """Kết xuất bảng rồi đọc lại phải ra đúng bảng cũ (phân tích nhiều đoạn, nạp phiên).

    Kiểm 3000 bảng ngẫu nhiên (hạt giống cố định) từ các mảnh mô tả kiểu model hay viết.
    Bản 3.23.613 hỏng 2483/3000: câu "Xưng hô: …"/"Sư phụ: …" trong mô tả làm rơi cả bảng
    xuống nhánh dự phòng, và ``[~70]`` (chưa rõ giới, có tuổi) đọc "~70" thành giới tính.
    """
    import random

    from subtitles_extractor.application.services.roster_format import (
        RosterEntry,
        render_roster,
    )

    rng = random.Random(1)
    names = ["叶然", "楚瑶", "师尊 / 师父", "王二婶", "Tiểu Vân", "A Tú", "林昆", "系统"]
    readings = ["Diệp Nhiên", "Sở Dao", "Sư tôn / Sư phụ", "", "Vương nhị thẩm"]
    genders = ["nam", "nữ", ""]
    ages = ["~20", "trung niên", "", "~70", "18"]
    pieces = [
        "Nhân vật chính", "đệ tử Linh Tiêu Tông; người xuyên không",
        "Xưng hô: với Sở Dao tự xưng 'huynh' gọi 'muội'.", "Quan hệ: vợ của A.",
        "Ghi chú: hay nói (đùa) [cười].", "vai trò — phản diện",
        "xưng hô: với 朱正軍 tự xưng 'tao' gọi 'mày'",
        "Tên thật: Lý A. Biệt danh: Tiểu A.",
        "Đệ tử. Sư phụ: Lão Vương. Tình trạng: đã chết.",
        "A: B; C: D.", "Nữ phụ: em gái.",
    ]
    broken = 0
    for _ in range(3000):
        entries: list[RosterEntry] = []
        for name in rng.sample(names, rng.randint(1, 6)):
            entries.append(RosterEntry(
                name=name, aka=rng.choice(readings), gender=rng.choice(genders),
                age=rng.choice(ages),
                role=" ".join(rng.sample(pieces, rng.randint(0, 3))),
            ))
        parsed = parse_roster(render_roster(entries))
        if [(e.name, e.aka, e.gender, e.age, e.role) for e in parsed] != [
            (e.name, e.aka, e.gender, e.age, e.role) for e in entries
        ]:
            broken += 1
    assert broken == 0
