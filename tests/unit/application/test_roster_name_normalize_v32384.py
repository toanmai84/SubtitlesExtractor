"""[v3.23.84] Test chuẩn hoá tên người nói theo roster (canonical name map).

Mục tiêu: tag người nói nhất quán xuyên tập — alias CJK -> tên Việt chuẩn, đồng nhất
hoa-thường/khoảng trắng — NHƯNG bảo thủ: KHÔNG gộp mờ các tên khác nhau.
"""

from __future__ import annotations

from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase as U,
)

_ROSTER = (
    "Hứa Phượng Niên (许凤年): nhân vật phụ: con trai Trấn Bắc Vương\n"
    "Hoàng đế (朕/皇帝): nhân vật chính: vua Đại Chu\n"
    "Thái hậu (太后): nhân vật phụ: mẹ vua\n"
)


class TestBuildCanonicalNameMap:
    def test_registers_canonical_and_cjk_aliases(self) -> None:
        cm = U._build_canonical_name_map(_ROSTER)
        assert cm["许凤年"] == "Hứa Phượng Niên"
        assert cm["hứa phượng niên"] == "Hứa Phượng Niên"
        assert cm["朕"] == "Hoàng đế"
        assert cm["皇帝"] == "Hoàng đế"
        assert cm["太后"] == "Thái hậu"

    def test_empty_roster_yields_empty_map(self) -> None:
        assert U._build_canonical_name_map("") == {}

    def test_handles_legacy_cjk_first_format(self) -> None:
        # Định dạng cũ "CJK (Việt)" vẫn nhận đúng phần Việt làm tên chuẩn.
        cm = U._build_canonical_name_map("林昆 (Lâm Côn): vai phụ")
        assert cm["林昆"] == "Lâm Côn"
        assert cm["lâm côn"] == "Lâm Côn"


class TestLocalizeOneNameWithCanonical:
    def test_cjk_alias_maps_to_canonical_vietnamese(self) -> None:
        cm = U._build_canonical_name_map(_ROSTER)
        assert U._localize_one_name("皇帝", None, cm) == "Hoàng đế"
        assert U._localize_one_name("许凤年", None, cm) == "Hứa Phượng Niên"

    def test_case_and_spacing_normalized_to_canonical(self) -> None:
        cm = U._build_canonical_name_map(_ROSTER)
        assert U._localize_one_name("hoàng đế", None, cm) == "Hoàng đế"
        assert U._localize_one_name("Thái  hậu", None, cm) == "Thái hậu"

    def test_name_not_in_roster_is_unchanged(self) -> None:
        # Bảo thủ: tên ngoài roster KHÔNG bị gộp/đổi.
        cm = U._build_canonical_name_map(_ROSTER)
        assert U._localize_one_name("Tần Chính", None, cm) == "Tần Chính"

    def test_no_canonical_map_keeps_legacy_behavior(self) -> None:
        # canonical_map=None -> hành vi cũ (Latin giữ nguyên).
        assert U._localize_one_name("John Smith", None, None) == "John Smith"


class TestSpeakerTagCanonicalization:
    def test_localize_speaker_uses_canonical(self) -> None:
        cm = U._build_canonical_name_map(_ROSTER)
        assert U._localize_speaker("皇帝", None, cm) == "Hoàng đế"

    def test_canonical_applies_within_channel_note(self) -> None:
        # Dạng "Tên (chú thích)" — phần tên được chuẩn hoá, chú thích giữ.
        cm = U._build_canonical_name_map(_ROSTER)
        out = U._localize_speaker("皇帝 (trên radio)", None, cm)
        assert out.startswith("Hoàng đế (")


class TestRedundantNameNoteDropped:
    """[v3.23.86] Bỏ chú thích ngoặc khi chỉ là echo của TÊN, không phải kênh thoại."""

    def test_duplicate_name_note_dropped(self) -> None:
        # Model hay xuất 'Tần Chính (Tần Chính)' -> tag không được lặp tên.
        assert U._localize_speaker("Tần Chính (Tần Chính)") == "Tần Chính"

    def test_cjk_echo_note_dropped(self) -> None:
        cm = U._build_canonical_name_map("Tần Chính (秦政): vai chính")
        assert U._localize_speaker("Tần Chính (秦政)", None, cm) == "Tần Chính"

    def test_alias_and_canonical_collapse(self) -> None:
        cm = U._build_canonical_name_map("Tần Chính (秦政): vai chính")
        assert U._localize_speaker("秦政 (Tần Chính)", None, cm) == "Tần Chính"

    def test_real_channel_note_preserved(self) -> None:
        # Chú thích kênh thoại THẬT vẫn được giữ và dịch.
        assert (
            U._localize_speaker("MAN (on computer)")
            == "Người đàn ông (trên máy tính)"
        )
        assert U._localize_speaker("ASTRONAUT (on radio)") == "Phi hành gia (qua radio)"


class TestCjkRoleFallback:
    """[v3.23.90] Fallback chức danh CJK -> Hán-Việt, tránh rò ký tự Trung vào tag."""

    def test_common_cjk_role_mapped(self) -> None:
        assert U._localize_one_name("大臣", None) == "Đại thần"
        assert U._localize_one_name("老奴", None) == "Lão nô"
        assert U._localize_one_name("太监", None) == "Thái giám"

    def test_slash_separated_cjk_uses_first_resolvable(self) -> None:
        assert U._localize_one_name("皇上/陛下", None) == "Hoàng thượng"

    def test_canonical_takes_priority_over_role_map(self) -> None:
        cm = U._build_canonical_name_map("Hoàng đế (朕/皇帝): vua")
        assert U._localize_one_name("皇帝", None, cm) == "Hoàng đế"

    def test_unknown_cjk_kept_unchanged(self) -> None:
        # Không có trong roster lẫn role map -> giữ nguyên (không bịa).
        assert U._localize_one_name("某某甲乙", None) == "某某甲乙"

    def test_latin_name_unaffected(self) -> None:
        assert U._localize_one_name("Tần Chính", None) == "Tần Chính"
