# ruff: noqa: RUF001, RUF002 — dấu câu fullwidth CJK thuộc câu gốc chép từ phim
"""[v3.23.613] Xưng hô thuần Việt không được cắt cụt tên riêng trùng âm hậu tố vai vế.

Tự rà lại bản 3.23.612: tên "Lâm Di" (林怡 — 怡 đọc "di") bị coi là "Lâm" + hậu tố
"di" (姨 = dì) khi câu gốc có chữ 姨 ở chỗ khác ("林怡，阿姨叫你呢") → "Chị Lâm, dì gọi
cậu kìa": mất tên, sai cả vai vế. Bảng nhân vật cũng tách nhầm như vậy nên âm tiết "di"
không được coi là tên ("Di tỷ" = 怡姐 không đổi được thành "chị Di").

Quy tắc: hậu tố VIẾT HOA chỉ là tiếng gọi khi bảng nhân vật xác nhận — người mang cụm
đó có tên chữ Hán kết thúc bằng đúng chữ Hán của hậu tố (王姨 → "Vương Di").
"""

from __future__ import annotations

from subtitles_extractor.application.services.native_address import (
    build_roster,
    name_syllables,
    nativize_line,
)

ROSTER = "\n".join([
    "林怡 (Lâm Di) [nữ, ~20] — con gái nhà họ Lâm; xưng hô: với 小東 tự xưng 'chị' gọi "
    "'em'",
    "小東 (Tiểu Đông) [nam, ~10] — em trai của Lâm Di; xưng hô: với 林怡 tự xưng 'em' "
    "gọi 'chị'",
    "王姨 (Vương Di) [nữ, ~50] — người giúp việc nhà họ Lâm",
    "林志遠 / 遠哥 (Lâm Chí Viễn / Viễn ca) [nam, ~30] — anh cả",
])


def _line(text: str, source: str) -> str:
    roster = build_roster(ROSTER)
    return nativize_line(text, source, roster=roster, speaker="小東", listener="林怡")[0]


class TestNameEndingInSuffixSound:
    def test_full_name_is_kept(self) -> None:
        assert _line("Lâm Di, dì gọi cậu kìa.", "林怡，阿姨叫你呢") == (
            "Lâm Di, dì gọi cậu kìa."
        )

    def test_name_kept_while_real_calque_converts(self) -> None:
        assert _line("Viễn ca, Lâm Di đâu?", "遠哥，林怡呢") == "Anh Viễn, Lâm Di đâu?"

    def test_capitalized_suffix_confirmed_by_cjk_name_converts(self) -> None:
        """王姨 ghi "Vương Di": chữ Hán kết thúc bằng 姨 → đúng là tiếng gọi."""
        assert _line("Vương Di, Lâm Di đâu rồi?", "王姨，林怡呢") == (
            "Dì Vương, Lâm Di đâu rồi?"
        )

    def test_name_syllable_is_kept_from_the_roster(self) -> None:
        assert "di" in name_syllables(ROSTER)

    def test_given_name_with_real_suffix_converts(self) -> None:
        assert _line("Di tỷ, em đói.", "怡姐，我餓了") == "Chị Di, em đói."
