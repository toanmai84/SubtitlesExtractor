"""Test [v3.23.55] Translation Memory: truy hồi, định dạng, suy khoá phim bộ."""

from __future__ import annotations

from subtitles_extractor.application.services.translation_memory import (
    TranslationMemoryEntry as E,
)
from subtitles_extractor.application.services.translation_memory import (
    derive_series_key,
    format_reference_block,
    retrieve_relevant,
)


class TestRetrieve:
    def test_finds_similar(self) -> None:
        tm = [E("林恒走进来", "Lâm Hằng đi vào"), E("再见", "Tạm biệt")]
        r = retrieve_relevant("林恒慢慢走进来", tm, top_k=2)
        assert r and r[0].target_text == "Lâm Hằng đi vào"

    def test_below_threshold_excluded(self) -> None:
        tm = [E("完全不相关的句子内容", "Câu hoàn toàn khác")]
        r = retrieve_relevant("林恒", tm, min_score=70.0)
        assert r == []

    def test_empty_inputs(self) -> None:
        assert retrieve_relevant("", [E("a", "b")]) == []
        assert retrieve_relevant("x", []) == []
        assert retrieve_relevant("x", [E("a", "b")], top_k=0) == []

    def test_respects_top_k(self) -> None:
        tm = [E(f"句子{i}", f"câu{i}") for i in range(10)]
        r = retrieve_relevant("句子1", tm, top_k=3, min_score=0.0)
        assert len(r) <= 3


class TestFormatBlock:
    def test_empty_returns_empty(self) -> None:
        assert format_reference_block([]) == ""

    def test_contains_pairs(self) -> None:
        block = format_reference_block([E("林恒", "Lâm Hằng")])
        assert "BỘ NHỚ DỊCH" in block
        assert "林恒" in block and "Lâm Hằng" in block

    def test_dedup_and_limit(self) -> None:
        entries = [E("a", "A"), E("a", "A"), E("b", "B")]
        block = format_reference_block(entries, max_entries=5)
        # 'a' chỉ xuất hiện 1 lần (dedup theo source).
        assert block.count('"a"') == 1


class TestDeriveSeriesKey:
    def test_unix_path(self) -> None:
        assert derive_series_key("/data/反派师尊/EP01.mp4") == "反派师尊"

    def test_windows_path(self) -> None:
        assert derive_series_key("D:\\Phim\\TuTien\\tap01.mkv") == "TuTien"

    def test_empty_or_shallow(self) -> None:
        assert derive_series_key("") == ""
        assert derive_series_key("movie.mp4") == ""

    def test_pathlib_input(self) -> None:
        """[v3.23.67] Hồi quy: tầng UI có thể truyền ``pathlib.Path`` (``WindowsPath``)

        chứ không phải ``str``. Trước đây ``.strip()`` trên ``Path`` gây
        ``AttributeError: 'WindowsPath' object has no attribute 'strip'``.
        """
        from pathlib import PurePosixPath, PureWindowsPath

        assert derive_series_key(PurePosixPath("/data/反派师尊/EP01.mp4")) == "反派师尊"
        win_path = PureWindowsPath("D:\\Phim\\TuTien\\tap01.mkv")
        assert derive_series_key(win_path) == "TuTien"
        # None vẫn an toàn (không biến thành chuỗi "None").
        assert derive_series_key(None) == ""  # type: ignore[arg-type]


class TestSeriesFromFilename:
    """[v3.23.82] Ưu tiên nhận diện series từ tên file qua mẫu SxxExx."""

    def test_extracts_series_before_marker(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            _series_name_from_filename,
        )

        assert _series_name_from_filename("NOVA.S51E16.Building.Stuff") == "NOVA"
        assert _series_name_from_filename("Show.Name.S01E02.720p") == "Show.Name"

    def test_no_marker_returns_empty(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            _series_name_from_filename,
        )

        assert _series_name_from_filename("just_a_movie") == ""
        # Mẫu ở đầu, không có phần tên trước -> rỗng (để fallback thư mục cha).
        assert _series_name_from_filename("S51E16") == ""

    def test_case_insensitive(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            _series_name_from_filename,
        )

        assert _series_name_from_filename("show.s01e02") == "show"


class TestDeriveSeriesKeyFilenameAware:
    """[v3.23.82] Tên file có SxxExx được ưu tiên — fix gộp nhầm ở gốc ổ đĩa."""

    def test_drive_root_uses_filename_not_drive_letter(self) -> None:
        # Trước đây trả "G:" (gộp mọi phim ở gốc ổ); nay trả đúng tên series.
        assert (
            derive_series_key("G:\\NOVA.S51E16.Building.Stuff.1080p.mkv") == "NOVA"
        )

    def test_filename_marker_beats_wrong_parent_folder(self) -> None:
        # Tệp trong thư mục "Downloads" nhưng tên có series -> dùng tên series.
        assert derive_series_key("D:\\Downloads\\NOVA.S51E16.mkv") == "NOVA"

    def test_episode_without_series_in_name_falls_back_to_parent(self) -> None:
        # Tên file chỉ có mã tập -> fallback thư mục cha (hành vi cũ giữ nguyên).
        assert derive_series_key("D:\\NOVA\\S51E16.mkv") == "NOVA"


class TestMergeCharacters:
    """[v3.23.91] Gộp roster xuyên tập (tích luỹ tên chuẩn), dedup theo danh tính."""

    def test_dedup_by_cjk_alias(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            merge_characters,
        )

        a = "Hứa Phượng Niên (许凤年): con trai"
        b = "Phượng Niên (许凤年): thế tử"
        merged = merge_characters(a, b)
        # Trùng alias CJK -> giữ mục cũ, không thêm trùng.
        assert merged == a

    def test_adds_new_characters(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            merge_characters,
        )

        a = "Hoàng đế (朕/皇帝): vua"
        b = "Hoàng đế (朕/皇帝): vua\nThái hậu (太后): mẹ vua"
        merged = merge_characters(a, b)
        assert "Hoàng đế" in merged
        assert "Thái hậu" in merged
        assert merged.count("Hoàng đế") == 1  # không nhân đôi

    def test_distinct_identities_both_kept(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            merge_characters,
        )

        a = "Hoàng đế (朕/皇帝): vua"
        b = "Chu Đế (周帝): hoàng đế"
        merged = merge_characters(a, b)
        # Danh tính khác nhau (không chung token) -> giữ CẢ hai (bảo thủ).
        assert "Hoàng đế" in merged and "Chu Đế" in merged

    def test_empty_inputs(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            merge_characters,
        )

        assert merge_characters("", "") == ""
        assert merge_characters("Vua (王): x", "") == "Vua (王): x"


class TestRosterOverlapRatio:
    """[v3.23.92] Phát hiện gộp nhầm phim bộ qua độ trùng roster."""

    def test_high_overlap_same_series(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            roster_overlap_ratio,
        )

        a = "Hoàng đế (朕): vua\nThái hậu (太后): mẹ"
        b = "Hoàng đế (朕): vua bá đạo\nNgụy Tài (魏才): thái giám"
        assert roster_overlap_ratio(a, b) >= 0.4  # chung "Hoàng đế/朕"

    def test_zero_overlap_different_series(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            roster_overlap_ratio,
        )

        a = "Tần Chính (秦政): tổng tài\nSở Nhan (楚颜): quý phi"
        b = "Lâm Côn (林昆): hoàng đế\nDuệ Thân Vương (睿亲王): hoàng đệ"
        assert roster_overlap_ratio(a, b) == 0.0

    def test_empty_roster_returns_zero(self) -> None:
        from subtitles_extractor.application.services.translation_memory import (
            roster_overlap_ratio,
        )

        assert roster_overlap_ratio("", "Vua (王): x") == 0.0
        assert roster_overlap_ratio("Vua (王): x", "") == 0.0
