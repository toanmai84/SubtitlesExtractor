"""[v3.23.637] Sửa tất định đại từ sai thế giới phim trong bảng nhân vật đã lưu.

Dữ liệu thật: ``series_context.characters`` của tệp chuẩn đoán dịch 86 tập (tu tiên): Diệp Nhiên tự xưng
'cháu' với sư thúc; Tần Chấp 'tôi'/'cậu'; Ma Viêm, Ma Hải 'tôi'; Diệp Nhiên 'tôi' với Thiên Thanh.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from subtitles_extractor.application.services.roster_format import address_rule_map
from subtitles_extractor.application.services.roster_register import (
    looks_ancient,
    normalize_roster_register,
)
from subtitles_extractor.application.use_cases.translate_subtitles import (
    TranslateSubtitlesUseCase,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationContext

OVERVIEW = (
    "Diệp Nhiên là đệ tử Tiểu Vân Phong, tu vi Đại Thừa nhưng bị hệ thống phong ấn; sư thúc Nam Cung "
    "Mộng, sư phụ, chưởng giáo, trưởng lão Linh Tiêu Tông và Ma Môn tranh đoạt cơ duyên tu luyện."
)
ROSTER = (
    "叶然 (Diệp Nhiên) [nam, ~22] — chính. Xưng hô: với 南宫梦 tự xưng 'cháu' gọi 'sư thúc', với 东方婉欣 "
    "tự xưng 'đồ nhi' gọi 'sư phụ', với 天青 xưng 'tôi' gọi 'ông / sư bá'.\n"
    "秦执 (Tần Chấp) [nam, ~23] — đối thủ; xưng hô: với 叶然 xưng 'tôi' gọi 'cậu'\n"
    "魔炎 (Ma Viêm) [nam, trung niên] — Ma Môn. Xưng hô: với 南宫梦 xưng 'tôi' gọi 'ngươi'.\n"
    "南宫梦 (Nam Cung Mộng) [nữ, ~28] — sư thúc; xưng hô: với 叶然 xưng 'ta' gọi 'ngươi'.\n"
    "东方婉欣 (Đông Phương Uyển Hân) [nữ, ~30] — sư phụ của Diệp Nhiên\n"
    "天青 (Thiên Thanh) [nam, ~40] — sư bá của Diệp Nhiên"
)


def test_real_roster_is_fixed_clause_by_clause() -> None:
    fixed, n = normalize_roster_register(ROSTER, OVERVIEW)
    assert n == 6
    rules = address_rule_map(fixed)
    assert rules[("叶然", "南宫梦")] == "tự xưng 'đệ tử' gọi 'sư thúc'"
    assert rules[("叶然", "东方婉欣")] == "tự xưng 'đồ nhi' gọi 'sư phụ'"      # đã đúng: giữ
    assert rules[("秦执", "叶然")] == "xưng 'ta' gọi 'ngươi'"
    assert rules[("魔炎", "南宫梦")] == "xưng 'ta' gọi 'ngươi'"
    assert rules[("南宫梦", "叶然")] == "xưng 'ta' gọi 'ngươi'"                   # bề trên: giữ 'ta'
    assert "xưng 'đệ tử' gọi 'ông / sư bá'" in fixed                              # 'tôi' → 'ta' → 'đệ tử'
    assert "tôi" not in fixed.lower().replace("tôi'", "x")  # không còn đại từ 'tôi' trong mệnh đề


def test_modern_film_is_untouched() -> None:
    modern = "宋钱 (Tống Tiền) [nam] — bác sĩ. Xưng hô: với 张依贝 xưng 'tôi' gọi 'cậu'"
    assert normalize_roster_register(modern, "Bác sĩ trẻ ở bệnh viện thành phố.") == (modern, 0)
    assert not looks_ancient(modern, "Bác sĩ trẻ ở bệnh viện thành phố.")


def test_idempotent_and_empty() -> None:
    fixed, _ = normalize_roster_register(ROSTER, OVERVIEW)
    assert normalize_roster_register(fixed, OVERVIEW) == (fixed, 0)
    assert normalize_roster_register("", OVERVIEW) == ("", 0)


def test_translate_use_case_applies_it_to_the_saved_context() -> None:
    @dataclass(frozen=True)
    class Req:
        context: TranslationContext

    request = Req(TranslationContext(target_lang="Vietnamese", source_lang="zh",
                                     characters=ROSTER, overview=OVERVIEW))
    out = TranslateSubtitlesUseCase._normalize_roster_register(request)
    assert out.context.characters != ROSTER and "'cháu'" not in out.context.characters
    assert out.context.overview == OVERVIEW
    clean = Req(replace(request.context, characters=out.context.characters))
    assert TranslateSubtitlesUseCase._normalize_roster_register(clean) is clean
