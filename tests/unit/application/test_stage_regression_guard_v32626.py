"""[v3.23.226] CHỐT CHẶN chống hồi quy: giai đoạn sau chỉ được GIỮ hoặc CẢI THIỆN.

Log chạy thật v225 cho thấy vòng tự sửa CÓ tác dụng — mọi dòng được gắn cờ
``needs_expansion`` đều được bổ sung — nhưng mỗi giai đoạn lại làm rớt nội dung ở một dòng
KHÁC vốn đang tốt::

    GĐ2 (dịch thô)     -> hỏng: 82, 71, 80
    GĐ3 (tinh chỉnh)   -> sửa xong 82/71/80, nhưng LÀM HỎNG 84
    GĐ4 (bản địa hoá)  -> sửa xong 84, nhưng LÀM HỎNG LẠI 82

Ta đuổi bắt vòng tròn: mỗi lượt tinh chỉnh vừa sửa lỗi cũ vừa gieo lỗi mới, nên số dòng
hỏng không bao giờ về 0 (5 -> 3 -> 1 -> 1).

Chốt chặn: dòng nào ĐANG BÌNH THƯỜNG ở đầu vào giai đoạn mà thành ĐÁNG NGỜ ở đầu ra thì
HOÀN NGUYÊN về bản trước. Bản trước có thể kém trau chuốt, nhưng KHÔNG mất nội dung — mà
mất nội dung là lỗi nặng hơn nhiều so với câu chữ chưa mượt.
"""

from __future__ import annotations

import pathlib

from subtitles_extractor.application.services.under_translation_guard import (
    find_regressions,
    find_under_translated,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)

# Nền khớp trung vị THẬT của phim trong log (~3.3 ký tự Việt / 1 ký tự CJK).
_NEN_GOC = "你今天来这里做什么"
_NEN_DICH = "Hôm nay cháu tới đây làm gì thế hả cháu"


def _nen4() -> list[tuple[int, str, str, str]]:
    return [(i, _NEN_GOC, _NEN_DICH, _NEN_DICH) for i in range(1, 30)]


def _nen3() -> list[tuple[int, str, str]]:
    return [(i, _NEN_GOC, _NEN_DICH) for i in range(1, 30)]


# ── find_regressions: chỉ bắt dòng bị LÀM TỆ ĐI ─────────────────────────────
def test_bat_dung_dong_bi_lam_hong() -> None:
    # Ca thật: GĐ3 sửa được 82 nhưng làm hỏng 84.
    ca = [
        (82, "自然会受到一些", "sẽ chịu,", "đương nhiên sẽ phải chịu một số"),
        (84, "所以我才跟着大家", "Nên cháu mới theo mọi người", "Nên cháu hùa,"),
    ]
    assert find_regressions(_nen4() + ca) == {84}


def test_khong_bat_dong_von_da_hong_tu_truoc() -> None:
    # Dòng hỏng SẴN ở đầu vào, vẫn hỏng ở đầu ra -> không phải hồi quy của giai đoạn này.
    # (Hoàn nguyên nó về bản cũ cũng chẳng ích gì — cả hai đều thiếu.)
    ca = [(82, "自然会受到一些", "sẽ chịu,", "sẽ chịu thôi,")]
    assert find_regressions(_nen4() + ca) == set()


def test_khong_bat_dong_duoc_cai_thien() -> None:
    ca = [(80, "作为王建强的侄女", "Là cháu chú,", "Là cháu gái của Vương Kiến Cường,")]
    assert find_regressions(_nen4() + ca) == set()


def test_khong_bat_dong_khong_doi() -> None:
    assert find_regressions(_nen4()) == set()


# ── Tái lập TOÀN BỘ diễn biến log v225 -> phải về 0 ─────────────────────────
def test_tai_lap_log_v225_va_ve_khong() -> None:
    goc = {
        82: "自然会受到一些",
        71: "你这么多年在外面",
        80: "作为王建强的侄女",
        84: "所以我才跟着大家",
    }
    gd2 = {
        82: "sẽ chịu,",
        71: "Bao năm qua,",
        80: "Là cháu chú,",
        84: "Nên cháu mới theo mọi người",
    }
    gd3 = {  # sửa được 3 dòng, nhưng làm hỏng 84
        82: "đương nhiên sẽ phải chịu một số",
        71: "Bao năm qua cháu ở bên ngoài,",
        80: "Là cháu gái của Vương Kiến Cường,",
        84: "Nên cháu hùa,",
    }
    gd4 = {  # sửa được 84, nhưng làm HỎNG LẠI 82
        82: "sẽ chịu,",
        71: "Bao năm qua cháu ở bên ngoài,",
        80: "Là cháu gái của Vương Kiến Cường,",
        84: "Nên cháu mới hùa theo mọi người",
    }

    def nghi(ban: dict[int, str]) -> list[int]:
        pairs = _nen3() + [(k, goc[k], v) for k, v in ban.items()]
        return sorted(s.index for s in find_under_translated(pairs))

    def chot_chan(truoc: dict[int, str], sau: dict[int, str]) -> dict[int, str]:
        items = _nen4() + [(k, goc[k], truoc[k], sau[k]) for k in sau]
        hong = find_regressions(items)
        return {k: (truoc[k] if k in hong else v) for k, v in sau.items()}

    # Không có chốt chặn: đuổi bắt vòng tròn, không bao giờ về 0 (đúng như log).
    assert nghi(gd2) == [71, 80, 82]
    assert nghi(gd3) == [84]
    assert nghi(gd4) == [82]

    # Có chốt chặn: về 0 ngay sau giai đoạn tinh chỉnh, và GIỮ NGUYÊN 0.
    sau3 = chot_chan(gd2, gd3)
    assert nghi(sau3) == []
    sau4 = chot_chan(sau3, gd4)
    assert nghi(sau4) == []
    # Dòng 82 giữ được bản ĐẦY ĐỦ của GĐ3, không bị GĐ4 phá.
    assert sau4[82] == "đương nhiên sẽ phải chịu một số"


# ── Tích hợp adapter ────────────────────────────────────────────────────────
def _line(idx: int, text: str, original: str) -> TranslationLine:
    return TranslationLine(
        index=idx, start_ms=idx * 2000, end_ms=idx * 2000 + 1500,
        text=text, original_text=original,
    )


def test_adapter_hoan_nguyen_dong_bi_lam_hong() -> None:
    dau_vao = [_line(i, _NEN_DICH, _NEN_GOC) for i in range(1, 30)]
    dau_vao.append(_line(84, "Nên cháu mới theo mọi người", "所以我才跟着大家"))
    dau_ra = [_line(i, _NEN_DICH, _NEN_GOC) for i in range(1, 30)]
    dau_ra.append(_line(84, "Nên cháu hùa,", "所以我才跟着大家"))  # bị làm hỏng

    ket_qua = GeminiSubtitleTranslator._revert_stage_regressions(
        dau_vao, dau_ra, use_dual_payload=True
    )
    dong_84 = next(ln for ln in ket_qua if ln.index == 84)
    assert dong_84.text == "Nên cháu mới theo mọi người"  # đã hoàn nguyên


def test_adapter_khong_dung_toi_giai_doan_khong_co_ban_goc() -> None:
    # Giai đoạn dịch thô: không có bản gốc để đối chiếu -> không được can thiệp.
    dau_vao = [_line(1, "a", "原文")]
    dau_ra = [_line(1, "b", "原文")]
    ket_qua = GeminiSubtitleTranslator._revert_stage_regressions(
        dau_vao, dau_ra, use_dual_payload=False
    )
    assert ket_qua[0].text == "b"


def test_co_ghi_log_khi_hoan_nguyen() -> None:
    src = pathlib.Path(
        "src/subtitles_extractor/infrastructure/translation/gemini_translation_adapter.py"
    ).read_text(encoding="utf-8")
    assert "HOÀN NGUYÊN về bản" in src
    assert "thà kém mượt còn hơn mất nghĩa" in src
