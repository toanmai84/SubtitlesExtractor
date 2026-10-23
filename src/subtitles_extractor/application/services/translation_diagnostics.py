"""[v3.23.77] Dựng "gói chẩn đoán dịch" — toàn bộ dữ liệu liên quan tới một lần dịch.

Mục đích: người dùng xuất một file JSON duy nhất (phụ đề gốc, kết quả dịch cuối, từng giai
đoạn dịch, phân tích ngữ cảnh: nhân vật/tóm tắt/glossary/visual cues, ngữ cảnh phim bộ,
tệp video cloud…) để gửi đi phân tích, khắc phục lỗi và cải thiện CHẤT LƯỢNG dịch.

Hàm ``build_diagnostics_bundle`` là HÀM THUẦN: nhận dữ liệu đã thu thập, trả dict
JSON-serializable; không I/O, không phụ thuộc Qt — dễ kiểm thử.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from itertools import pairwise
from typing import Any, Protocol


class _EventLike(Protocol):
    index: int
    text: str

    @property
    def start_sec(self) -> float: ...

    @property
    def end_sec(self) -> float: ...


_SCHEMA = "subtitles_extractor.translation_diagnostics"
_SCHEMA_VERSION = 3

# [v3.23.83] Calibrate cờ length_anomaly dựa trên dữ liệu thật (zh->vi): tiếng Trung cực
# ngắn gọn (1 Hán tự ~ 1 từ) nên so độ dài thô với tiếng Việt phồng giả tạo. Khắc phục:
#   1. Bỏ tag người nói "[Tên:]" (chỉ có ở bản dịch) trước khi đo.
#   2. Quy đổi độ dài nguồn CJK theo hệ số giãn (1 ký tự CJK ~ vài ký tự Latinh/Việt).
#   3. Bỏ qua nguồn quá ngắn (tỉ lệ không đáng tin với câu 1-2 ký tự).
_SPEAKER_TAG_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_CJK_EXPANSION = 3.0  # 1 ký tự CJK quy đổi ~ 3 ký tự Latinh/Việt

#: [v3.23.438] Ngưỡng cờ tốc độ đọc. CPS (ký tự/giây) > ngưỡng này là khó đọc kịp cho phụ đề
#: tiếng Việt (chuẩn ngành ~17-21 CPS; lấy 25 để chỉ khoanh ca RÕ RÀNG quá nhanh, tránh nhiễu).
_READING_SPEED_MAX_CPS = 25.0
#: Thời lượng tối thiểu (giây) để tin timing — dưới mức này thường là lỗi timing OCR (hiển thị
#: 0.2-0.5s phi thực tế) chứ không phải dịch dài → bỏ qua để tránh dương tính giả.
_READING_SPEED_MIN_DUR = 0.8

#: [v3.23.444] Model thời lượng đọc TTS tiếng Việt (đo thực VieNeu): dur ≈ BASE + PER_CHAR × n.
#: Dùng để ước lượng NÉN audio: ratio = dur_tự_nhiên / thời_lượng_hiển_thị. >1.6x = "tiếng gió"
#: (WSOLA làm formant tan khi nén mạnh) → TTS chói/méo. Đây là mặt TTS của cùng vấn đề text-quá-
#: dài-so-với-thời-lượng mà reading_speed đo ở mặt MẮT ĐỌC.
_TTS_DURATION_BASE_S = 0.302
_TTS_DURATION_PER_CHAR_S = 0.0486
_TTS_COMPRESSION_MAX_RATIO = 1.6


#: [v3.23.557] Dấu câu hai bảng mã (ASCII và full-width) — hardsub trộn lẫn cả hai.
_PUNCTUATION_RE = re.compile(
    r"[\s\u3000!-/:-@\[-`{-~\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff5e"
    r"\u3001\u3002\u300c\u300d\u2026\u00b7\u2018\u2019\u201c\u201d]+"
)


def strip_punctuation(text: str) -> str:
    """[v3.23.557] Bỏ MỌI dấu câu và khoảng trắng để so hai câu theo NỘI DUNG.

    Vì sao cần: phụ đề hardsub trộn dấu câu ASCII với full-width. Ca thật v3.23.556: hai
    dòng gốc "Tống Tiền" viết với dấu chấm than ở HAI bảng mã khác nhau bị coi là "nguồn
    khác nhau", nên bản dịch giống hệt (vốn đúng) lại bị báo trùng — tầng dịch còn tốn
    thêm một lượt gọi để dịch lại hai dòng đã đúng.

    Args:
        text: Câu cần chuẩn hoá.

    Returns:
        Chuỗi chỉ còn chữ và số, viết thường.
    """
    return _PUNCTUATION_RE.sub("", (text or "")).lower()


#: Trợ từ ngữ khí tiếng Trung — không mang nội dung ("怎么是你啊" = "怎么是你").
_MODAL_PARTICLE_RE = re.compile("[啊呀吧呢嘛哦哎啦吗嗎么哈喔噢哇呐咯嘞]")


def source_content_key(source: str) -> str:
    """[v3.23.580] Nội dung câu gốc để so trùng: bỏ dấu câu và trợ từ ngữ khí (hàm thuần).

    Dùng chung cho cờ chuẩn đoán ``duplicate_adjacent`` và khâu dọn dòng trùng — trước
    đây chỉ khâu dọn bỏ trợ từ (v3.23.577), nên "住手" / "住手啊" cùng ra "Dừng tay!"
    (dịch ĐÚNG, khâu dọn không đụng) vẫn bị file chuẩn đoán báo trùng.

    Args:
        source: Câu gốc.

    Returns:
        Chuỗi nội dung đã chuẩn hoá.
    """
    return _MODAL_PARTICLE_RE.sub("", strip_punctuation(source))


#: [v3.23.602] Tag trong ngoặc vuông ("[cười]", "[Tên:]") — phụ đề xuất ra không có.
_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]")


def _normalize_for_consistency(text: str) -> str:
    """[v3.23.442] Chuẩn hoá bản dịch để so nhất quán: bỏ dấu câu cuối + khoảng trắng + hoa
    đầu câu — để chỉ còn khác biệt THẬT (từ ngữ/thứ tự) mới bị coi là không nhất quán.

    [v3.23.602] Bỏ cả tag trong ngoặc vuông: "[cười] Rượu gặp tri kỷ" và "Rượu gặp tri
    kỷ..." (dòng 455/454 chuẩn đoán thật) là CÙNG một câu — tag cảm xúc chỉ vào giọng
    đọc, bộ xuất SRT/ASS đã lọc nó khỏi phụ đề.
    """
    untagged = _BRACKET_TAG_RE.sub(" ", text or "")
    stripped = " ".join(untagged.split()).strip(".,!?;:…\"'-—)( ").strip()
    return stripped.lower()
# [v3.23.156] 8 -> 12: câu cực ngắn ("Yes, I am." -> "Có.") có tỉ lệ vô nghĩa.
_MIN_EFFECTIVE_LEN = 12.0  # bỏ qua nguồn hiệu dụng ngắn hơn ngưỡng này
# [v3.23.156] 0.5 -> 0.35: bản dịch NGẮN có chủ đích là tính năng (quy tắc "súc tích
# cho TTS" trong prompt); dữ liệu thật (The Hot Spot) cho thấy ratio 0.4-0.5 vẫn là
# bản dịch tốt. Chỉ cảnh báo khi mất quá 2/3 nội dung (thật sự khả nghi nuốt ý).
_LENGTH_RATIO_LOW = 0.35
_LENGTH_RATIO_HIGH = 2.5
#: [v3.23.404] Nguồn CJK NGẮN (eff < ngưỡng này ~ ≤6 ký tự CJK, hay là thành ngữ 4 chữ) nở ra
#: nhiều hơn là bình thường → dùng cận trên NỚI để tránh dương tính giả với thành ngữ.
_SHORT_SOURCE_EFF = 18.0
_LENGTH_RATIO_HIGH_SHORT = 3.6
# Token coi là "không dịch được": số/dấu, hoặc từ viết Hoa chữ đầu (tên riêng).
_PROPER_TOKEN_RE = re.compile(r"^(?:[A-Z][\w'-]*|[0-9]+[\w.]*)$")
_WORD_SPLIT_RE = re.compile(r"[\s,.!?;:…\-—\"'()\[\]]+")


def _strip_speaker_tag(text: str) -> str:
    """Bỏ tiền tố tag người nói ``[Tên:]`` (chỉ xuất hiện ở bản dịch)."""
    return _SPEAKER_TAG_RE.sub("", text).strip()


def _effective_source_length(text: str) -> float:
    """Độ dài nguồn 'hiệu dụng': ký tự CJK nhân hệ số giãn, ký tự khác đếm 1.

    Nhờ đó so độ dài giữa nguồn CJK (rất đặc) và bản dịch Latinh/Việt công bằng hơn.

    [v3.23.156] Nguồn SONG NGỮ (hardsub 2 dòng CÙNG NGHĨA: một dòng CJK + một dòng
    Latin — vd "∮须得别处觅芳草∮" + "∮ Gotta find my baby ∮"): cộng cả hai làm độ
    dài nguồn PHỒNG ĐÔI -> bản dịch (chỉ dịch MỘT nghĩa) bị flag ngắn oan. Khi phát
    hiện đồng thời dòng CJK-dominant và dòng non-CJK, lấy MAX theo dòng thay vì tổng.
    """

    def _one_line(line: str) -> tuple[float, bool]:
        cjk = sum(1 for ch in line if _CJK_RE.match(ch))
        effective = cjk * _CJK_EXPANSION + (len(line) - cjk)
        stripped = line.strip()
        is_cjk_dominant = bool(stripped) and cjk >= max(1, len(stripped) // 2)
        return effective, is_cjk_dominant

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        measured = [_one_line(ln) for ln in lines]
        has_cjk_line = any(dom for _, dom in measured)
        has_latin_line = any(not dom for _, dom in measured)
        if has_cjk_line and has_latin_line:
            return max(eff for eff, _ in measured)
    effective, _ = _one_line(text)
    return effective


def _is_untranslatable_source(source_text: str) -> bool:
    """[v3.23.156] True nếu nguồn 'không có gì để dịch' (tên riêng/số/dấu câu).

    Dữ liệu thật (The Hot Spot): 25/25 cờ identical_to_source đều là các dòng như
    "Harry.", "Gloria...", "357...", "Madox. Harry Madox." — dịch GIỮ NGUYÊN là ĐÚNG.
    Tiêu chí: <= 4 token và MỌI token đều là tên riêng (Hoa chữ đầu) hoặc số.
    """
    tokens = [t for t in _WORD_SPLIT_RE.split(source_text) if t]
    if not tokens:
        return True  # toàn dấu câu/ký hiệu
    if len(tokens) > 4:
        return False
    return all(_PROPER_TOKEN_RE.match(token) for token in tokens)


#: [v3.23.549] Loại từ / danh từ đơn vị tiếng Việt — biến thiên theo sắc thái nên KHÔNG được coi
#: là một phần bắt buộc của thuật ngữ. Đo trên phim thật: phần lớn "vi phạm" còn sót chỉ là đổi
#: loại từ ("kẻ quê mùa" → "gã/tên quê mùa", "tiền sính lễ" → "sính lễ", "mảnh" → "miếng").
_VIETNAMESE_CLASSIFIERS = frozenset({
    "kẻ", "gã", "tên", "thằng", "người", "đứa", "con", "chiếc", "cái", "đôi", "cặp",
    "mảnh", "miếng", "tiền", "phí", "bệnh", "xe", "món", "bộ", "cuộc", "việc", "sự",
    "quả", "viên", "tấm", "ngôi", "căn", "toà", "tòa", "chuyện", "lễ", "buổi", "trận",
    "loại", "thứ", "vị", "bọn", "lũ", "nhà", "của", "một",
})

_CONTENT_TOKEN_RE = re.compile(r"[0-9a-zà-ỹđ]+", re.IGNORECASE)

#: Tỉ lệ TỪ NỘI DUNG của phương án phải xuất hiện thì mới coi là đạt.
#: Đo trên phim thật: 1.0 → 7 lượt vi phạm, 0.75 → 7, 0.6 → 5, 0.5 → 1 (quá dễ dãi, che lỗi thật).
#: Chọn 0.6: khử 88% nhiễu mà vẫn giữ nguyên các ca đáng rà tay.
_GLOSSARY_CONTENT_COVERAGE = 0.6

#: Số dòng lân cận cần soi hai bên. Dòng phụ đề là MẢNH CÂU và trật tự Hoa-Việt khác nhau, nên
#: thuật ngữ thường rơi sang dòng kề. Đo trên phim thật: ±1 khử 40% nhiễu, ±2 không thêm gì.
_GLOSSARY_NEIGHBOUR_LINES = 1


def _content_tokens(phrase: str) -> list[str]:
    """Các từ NỘI DUNG của một cụm tiếng Việt (bỏ loại từ, bỏ token 1 ký tự)."""
    return [
        token
        for token in _CONTENT_TOKEN_RE.findall(phrase.lower())
        if token not in _VIETNAMESE_CLASSIFIERS and len(token) >= 2
    ]


def matches_expected_term(alternatives: list[str], haystack: str) -> bool:
    """Bản CÔNG KHAI của :func:`_matches_expected_term` — dùng chung cho lưới chặn hồi quy."""
    return _matches_expected_term(alternatives, haystack)


def _matches_expected_term(alternatives: list[str], haystack: str) -> bool:
    """[v3.23.549] True nếu ``haystack`` thể hiện thuật ngữ, chấp nhận biến thể loại từ.

    Khớp nguyên văn trước; nếu trượt thì so theo TỪ NỘI DUNG, vì tiếng Việt bắt buộc có loại từ
    và loại từ đổi theo sắc thái câu ("kẻ" / "gã" / "tên" quê mùa đều đúng).

    Args:
        alternatives: Các phương án dịch đã chuẩn hoá chữ thường.
        haystack: Văn bản cần tìm (đã chữ thường).

    Returns:
        True nếu có phương án nào đạt.
    """
    for alternative in alternatives:
        if alternative in haystack:
            return True
        tokens = _content_tokens(alternative)
        if not tokens:
            continue
        hits = sum(1 for token in tokens if token in haystack)
        if hits / len(tokens) >= _GLOSSARY_CONTENT_COVERAGE:
            return True
    return False


GLOSSARY_KIND_NAME = "name"
"""Mục thuật ngữ là TÊN RIÊNG (người, địa danh, tổ chức, chiêu thức) — bắt buộc nhất quán."""

GLOSSARY_KIND_EXPRESSION = "expression"
"""Mục là khái niệm/thành ngữ/khẩu ngữ — diễn đạt khác theo ngữ cảnh vẫn có thể ĐÚNG."""


def glossary_alternatives(translation: str) -> list[str]:
    """[v3.23.550] Các phương án dịch hợp lệ của một mục thuật ngữ (chữ thường).

    Tách phần trong ngoặc TRƯỚC rồi mới tách theo ``/``. Làm ngược lại (như trước) thì dấu ``/``
    NẰM TRONG ngoặc cắt đôi cặp ngoặc: ``"Thổ Cẩu (chó ta/chó cỏ)"`` thành hai mẩu
    ``"thổ cẩu (chó ta"`` và ``"chó cỏ)"``, ngoặc hở không bị bỏ nên KHÔNG mẩu nào khớp được —
    bản dịch "Thổ Cẩu" hoàn toàn đúng bị tính vi phạm 4/6 trong file chuẩn đoán thật.

    Nội dung trong ngoặc cũng là phương án hợp lệ (``大G = "xe Mercedes G-Class (Đại G)"``: dùng
    "Đại G" là đúng), và cũng có thể chứa nhiều phương án ngăn bằng ``/``.

    Args:
        translation: Vế dịch của mục thuật ngữ.

    Returns:
        Danh sách phương án, bỏ trùng, giữ thứ tự.
    """
    notes = re.findall(r"\(([^)]*)\)", translation or "")
    base = re.sub(r"\([^)]*\)", " ", translation or "")
    candidates = [part for part in base.split("/")]
    for note in notes:
        candidates.extend(note.split("/"))
    result: list[str] = []
    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", candidate).strip().lower()
        if len(cleaned) >= 2 and cleaned not in result:
            result.append(cleaned)
    # [v3.23.552] TỪ GHÉP GẠCH NỐI trong phương án cũng là dạng rút gọn hợp lệ: "xe Mercedes
    # G-Class" → "G-Class". Bộ so theo từ nội dung tách "g-class" thành "g" + "class" rồi bỏ "g"
    # (quá ngắn) nên chỉ phủ 50% — chuẩn đoán thật báo vi phạm tên riêng oan cho "xe G-Class".
    for candidate in list(result):
        for word in candidate.split():
            if "-" in word and len(word) >= 5 and word not in result:
                result.append(word)
    return result


def glossary_entry_kind(translation: str) -> str:
    """[v3.23.550] Phân loại mục thuật ngữ: TÊN RIÊNG hay BIỂU ĐẠT.

    Vì sao cần: soi file chuẩn đoán thật, gần như mọi "vi phạm" còn lại là thành ngữ/khẩu ngữ
    (``作威作福``, ``虎视眈眈``, ``放屁``…) được dịch TỰ NHIÊN theo ngữ cảnh ("hống hách bá đạo",
    "nhòm ngó như hổ đói"). Ép đúng chữ trong bảng sẽ làm bản dịch gượng hơn, nên trộn chúng với
    vi phạm tên riêng khiến con số tổng mất ý nghĩa.

    Tiêu chí: phương án chính có chữ viết HOA ở từ KHÔNG đứng đầu (``"Tập đoàn Liễu Thị"``), hoặc
    mọi từ đều viết hoa (``"Hồng Liên Nữ Đế"``). Chữ hoa chỉ ở từ đầu là kiểu viết câu, không phải
    tên riêng (``"Thức hải"``).

    Args:
        translation: Vế dịch của mục thuật ngữ.

    Returns:
        :data:`GLOSSARY_KIND_NAME` hoặc :data:`GLOSSARY_KIND_EXPRESSION`.
    """
    primary = re.sub(r"\([^)]*\)", " ", translation or "").split("/")[0]
    words = [word for word in re.findall(r"\w+", primary) if not word.isdigit()]
    if not words:
        return GLOSSARY_KIND_EXPRESSION
    capitalised = [word[0].isupper() for word in words]
    if all(capitalised) or any(capitalised[1:]):
        return GLOSSARY_KIND_NAME
    return GLOSSARY_KIND_EXPRESSION


def _glossary_violations(
    source_by_index: dict[int, str],
    translated_list: list[_EventLike],
    glossary: str,
) -> list[dict[str, Any]]:
    """[v3.23.500] Đo mức TUÂN THỦ bảng thuật ngữ trong bản dịch cuối.

    Vì sao cần: glossary được dựng công phu nhưng khâu dịch có thể bỏ qua. Đo trên một bản dịch
    thật cho thấy 72/113 thuật ngữ bị dịch khác quy định (173 lượt), trong đó có ca mất hẳn nội
    dung. Không đo thì không biết để sửa.

    Mục glossary có thể ghi NHIỀU phương án ngăn bằng ``/`` (vd ``"thẻ căn cước / CMND"``) —
    dùng bất kỳ phương án nào cũng tính là đạt. Phần chú thích trong ngoặc được bỏ khi so khớp.

    Args:
        source_by_index: Bản đồ chỉ số → câu gốc.
        translated_list: Các sự kiện đã dịch.
        glossary: Bảng thuật ngữ, mỗi dòng ``gốc = bản dịch``.

    Returns:
        Danh sách mục bị vi phạm, sắp xếp giảm dần theo số lượt, kèm ví dụ dòng để soi nhanh.
    """
    if not glossary.strip():
        return []
    from subtitles_extractor.application.services.roster_format import parse_glossary

    from subtitles_extractor.application.services.han_variants import fold_han_variants
    from subtitles_extractor.application.services.term_variants import VariantIndex

    translated_by_index = {int(e.index): (e.text or "") for e in translated_list}
    findings: list[dict[str, Any]] = []
    entries = parse_glossary(glossary)
    variant_index = VariantIndex(fold_han_variants("\n".join(source_by_index.values())))
    known_terms = frozenset(
        fold_han_variants(alias.strip())
        for entry in entries
        for alias in re.split(r"\s*[/|]\s*", entry.term)
        if alias.strip()
    )
    for entry in entries:
        term = entry.term.strip()
        if len(term) < 2 or not entry.translation.strip():
            continue
        alternatives = glossary_alternatives(entry.translation)
        # [v3.23.507] Chấp nhận DẠNG RÚT GỌN hợp lệ của tên ghép bằng gạch nối: bảng ghi
        # "Mercedes-Benz" nhưng bản dịch dùng "Mercedes" là hoàn toàn đúng — trước đây bị tính
        # vi phạm 9/9 và còn bị gắn nhãn "mục glossary đáng ngờ" oan.
        for alternative in list(alternatives):
            core = alternative.split("-", 1)[0].strip()
            if len(core) >= 4 and core != alternative:
                alternatives.append(core)
        if not alternatives:
            continue
        # [v3.23.508] Vế TRÁI có thể gộp nhiều BÍ DANH ("满月宴 / 满月酒"): phải tìm TỪNG bí danh
        # trong phụ đề, không tìm nguyên chuỗi — nếu không, toàn bộ mục dạng này không bao giờ
        # khớp và bị BỎ QUA khi đo tuân thủ (dữ liệu thật: 21 mục quan trọng bị bỏ sót).
        term_aliases = [a.strip() for a in re.split(r"\s*[/|]\s*", term) if a.strip()]
        # [v3.23.645] Tên tông môn viết lệch một chữ cũng phải mang cùng bản dịch.
        variant_aliases: list[str] = []
        for alias in term_aliases:
            variant_aliases.extend(variant_index.variants_of(fold_han_variants(alias), known_terms))
        occurrences = 0
        violations: list[int] = []
        for index, source_text in source_by_index.items():
            if not (
                any(alias in source_text for alias in term_aliases)
                or (variant_aliases and any(
                    alias in fold_han_variants(source_text) for alias in variant_aliases
                ))
            ):
                continue
            occurrences += 1
            # [v3.23.549] Soi CẢ dòng kề: phụ đề hardsub là MẢNH CÂU, mà trật tự từ Hoa-Việt khác
            # nhau nên thuật ngữ rất hay rơi sang dòng bên. Ca thật: ``和江城医院的`` dịch "cùng
            # với Viện trưởng Tôn" còn "Bệnh viện Giang Thành" nằm ở dòng kế — bản dịch ĐÚNG mà bị
            # tính vi phạm. Đo trên phim thật: chỉ riêng thay đổi này khử 40% số lượt báo sai.
            window = " ".join(
                translated_by_index.get(neighbour, "").lower()
                for neighbour in range(
                    index - _GLOSSARY_NEIGHBOUR_LINES, index + _GLOSSARY_NEIGHBOUR_LINES + 1
                )
            )
            if not _matches_expected_term(alternatives, window):
                violations.append(index)
        if violations:
            findings.append({
                "term": term,
                "kind": glossary_entry_kind(entry.translation),
                "expected": entry.translation.strip(),
                "occurrences": occurrences,
                "violations": len(violations),
                "example_indices": violations[:5],
            })
    # Tên riêng lên trước: đó là loại vi phạm BẮT BUỘC phải sửa (đổi tên giữa chừng làm người xem
    # tưởng là nhân vật/địa danh khác); biểu đạt thường chỉ là lựa chọn diễn đạt.
    findings.sort(
        key=lambda item: (item["kind"] != GLOSSARY_KIND_NAME, -item["violations"])
    )
    return findings


SUSPECT_ENTRY_MIN_OCCURRENCES = 5
"""Số lần xuất hiện tối thiểu để kết luận một MỤC glossary là đáng ngờ (không phải ngẫu nhiên)."""

SUSPECT_ENTRY_VIOLATION_RATIO = 0.8
"""Vi phạm từ mức này trở lên thì lỗi nằm ở MỤC glossary, không phải ở bản dịch.

[v3.23.514] Hạ từ 0.9 xuống 0.8 sau khi soi dữ liệu thật: mục ``到账`` quy định cả CỤM DÀI
"tiền vào tài khoản (chuyển khoản thành công)" bị vi phạm 18/21 = 86%. Các bản dịch thực tế đều
hợp lý ("hai vạn tệ đã vào tài khoản") nhưng không chứa trọn cụm — đây là mục quy định quá dài,
không phải khâu dịch sai. Ngưỡng 0.9 để lọt nhóm này và thổi phồng số vi phạm."""


def split_suspect_entries(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """[v3.23.506] Tách "mục glossary ĐÁNG NGỜ" khỏi "vi phạm THẬT".

    Dữ liệu thật: mục ``小可爱`` yêu cầu dịch thành "tiểu khả ái" bị vi phạm 37/37 dòng. Khi bản
    dịch NHẤT QUÁN chọn cách khác ở gần như mọi lần xuất hiện thì lỗi nằm ở MỤC GLOSSARY (bản dịch
    quy định gượng ép), chứ không phải khâu dịch làm sai. Gộp chung hai loại khiến con số mất ý
    nghĩa và không biết nên sửa ở đâu.

    Args:
        findings: Kết quả từ :func:`_glossary_violations`.

    Returns:
        Cặp ``(vi_pham_that, muc_dang_ngo)``.
    """
    real: list[dict[str, Any]] = []
    suspect: list[dict[str, Any]] = []
    for item in findings:
        occurrences = int(item.get("occurrences") or 0)
        violations = int(item.get("violations") or 0)
        if (
            occurrences >= SUSPECT_ENTRY_MIN_OCCURRENCES
            and violations >= occurrences * SUSPECT_ENTRY_VIOLATION_RATIO
        ):
            suspect.append({**item, "hint": "bản dịch quy định có thể gượng ép — nên sửa mục này"})
        else:
            real.append(item)
    return real, suspect


MAX_SHIFT_DISTANCE = 5
"""Khoảng lệch tối đa cần dò — quan sát thực tế cho thấy lệch khối thường trong vòng 5 dòng."""


def _shifted_lines(
    source_by_index: dict[int, str],
    translated_list: list[_EventLike],
    glossary: str,
    characters: str,
) -> list[dict[str, Any]]:
    """[v3.23.515/523] Phát hiện dòng có bản dịch THUỘC VỀ DÒNG KHÁC (lệch dòng/lệch khối).

    [v3.23.523] Uỷ quyền hoàn toàn cho ``alignment_check`` — nguồn chân lý duy nhất, dùng chung với
    bộ kiểm chạy lúc dịch. Trước đây module này có BẢN SAO logic riêng, nên các cải tiến thêm vào
    service (miễn trừ câu tách đôi, lọc mốc neo quá phổ biến) không hề tác dụng lên cờ chuẩn đoán —
    đúng kiểu trùng lặp từng gây lỗi ở v3.23.491.

    [v3.23.549] Thêm hai bộ lọc chống dương tính giả cho đường MỐC NEO (bỏ neo có vế dịch phổ biến
    hơn vế gốc; đòi dòng bổ trợ cùng độ lệch). Trên phim thật cả 5 dòng bị cờ đều là dương tính
    giả và bị khử sạch.

    Args:
        source_by_index: Bản đồ chỉ số → câu gốc.
        translated_list: Các sự kiện đã dịch.
        glossary: Bảng thuật ngữ.
        characters: Bảng nhân vật.

    Returns:
        Danh sách dòng nghi lệch, kèm dòng mà nội dung thuộc về và khoảng lệch.
    """
    from subtitles_extractor.application.services.alignment_check import (
        build_anchor_pairs,
        corroborated_findings,
        drop_unreliable_anchors,
        find_misaligned_lines,
    )

    anchors = build_anchor_pairs(characters, glossary)
    if not anchors:
        return []
    translations = {int(event.index): (event.text or "") for event in translated_list}
    anchors = drop_unreliable_anchors(anchors, source_by_index, translations)
    if not anchors:
        return []
    return corroborated_findings(find_misaligned_lines(source_by_index, translations, anchors))


def _misaligned_blocks(
    source_by_index: dict[int, str], translated_list: list[_EventLike]
) -> list[dict[str, Any]]:
    """[v3.23.549] Tìm các KHỐI dòng nghi lệch bằng tương quan độ dài trên cửa sổ trượt.

    Bổ khuyết cho :func:`_shifted_lines`: mốc neo chỉ phủ khoảng 9% số dòng nên khối lệch không
    chứa tên riêng nào sẽ lọt hoàn toàn. Ca thật ở phim 3902 dòng: khối 2682-2707 sai toàn bộ 18
    dòng (model gộp hai dòng ngắn rồi đẩy lệch) mà KHÔNG cờ nào báo — cờ này bắt được.

    Args:
        source_by_index: Bản đồ chỉ số → câu gốc.
        translated_list: Các sự kiện đã dịch.

    Returns:
        Danh sách khối nghi lệch, nặng nhất trước.
    """
    from subtitles_extractor.application.services.alignment_check import (
        find_low_correlation_blocks,
    )

    translations = {int(event.index): (event.text or "") for event in translated_list}
    return find_low_correlation_blocks(source_by_index, translations)


def _number_issues(
    source_by_index: dict[int, str], translated_by_index: dict[int, str]
) -> list[dict[str, Any]]:
    """[v3.23.556] Con số sai giá trị hoặc lệch lối đọc so với phần còn lại của phim."""
    from subtitles_extractor.application.services.number_consistency_guard import (
        find_number_findings,
    )

    return [
        {
            "index": item.index,
            "source": item.source_expression,
            "expected_value": item.expected_value,
            "found": item.found_phrase,
            "found_value": item.found_value,
            "kind": item.kind,
        }
        for item in find_number_findings(source_by_index, translated_by_index)
    ]


def _outdated_address_rules(
    source_by_index: dict[int, str],
    translated_by_index: dict[int, str],
    visual_cues: str,
    characters: str,
) -> list[dict[str, Any]]:
    """[v3.23.557] Quy tắc xưng hô trong bảng nhân vật mà CẢ PHIM dùng khác hẳn.

    Đây là lỗi của BẢNG NHÂN VẬT chứ không phải của bản dịch: sửa một dòng roster là
    xong, thay vì soi tay hàng chục dòng. Các dòng thuộc nhóm này được rút khỏi
    ``address_rule_violations`` để danh sách đó chỉ còn ca thật sự cần soi.
    """
    from subtitles_extractor.application.services.address_rule_guard import (
        find_outdated_rules,
    )

    if not (visual_cues or "").strip() or not (characters or "").strip():
        return []
    return [
        {
            "speaker": rule.speaker,
            "addressee": rule.addressee,
            "rule_says": rule.expected,
            "film_uses": rule.observed,
            "lines_film": rule.lines_observed,
            "lines_rule": rule.lines_following,
            "first_index": rule.first_index,
            "suggestion": (
                f"Sửa bảng nhân vật: {rule.speaker} với {rule.addressee} "
                f"tự xưng '{rule.observed}' (phim dùng {rule.lines_observed} dòng) "
                f"thay vì '{rule.expected}'."
            ),
        }
        for rule in find_outdated_rules(
            source_by_index, translated_by_index, visual_cues, characters
        )
    ]


def _without_self_address(cues_json: str) -> str:
    """[v3.23.577] Bỏ các cue có người nói trùng người nghe khỏi JSON cues (hàm thuần)."""
    from subtitles_extractor.application.services.monologue_guard import is_self_addressed

    try:
        parsed = json.loads(cues_json) if (cues_json or "").strip() else []
    except (json.JSONDecodeError, TypeError):
        return cues_json
    if not isinstance(parsed, list):
        return cues_json
    kept = [
        item
        for item in parsed
        if not (
            isinstance(item, dict)
            and is_self_addressed(str(item.get("spk") or ""), str(item.get("to") or ""))
        )
    ]
    return json.dumps(kept, ensure_ascii=False)


def _monologue_self_forms(
    visual_cues: str,
    source_by_index: dict[int, str],
    translated_by_index: dict[int, str],
    characters: str,
) -> list[dict[str, Any]]:
    """[v3.23.577] Độc thoại tự xưng bằng từ vai vế (em/anh/chị…), xem monologue_guard."""
    from subtitles_extractor.application.services.monologue_guard import (
        monologue_self_forms,
    )
    from subtitles_extractor.application.services.visual_cue_serializer import (
        deserialize_visual_cues,
    )

    if not (visual_cues or "").strip():
        return []
    return monologue_self_forms(
        deserialize_visual_cues(visual_cues), source_by_index, translated_by_index,
        characters,
    )


def _echo_lines(
    source_by_index: dict[int, str], translated_by_index: dict[int, str]
) -> list[dict[str, Any]]:
    """[v3.23.576] Dòng dịch NÓI LẶP dòng kề mà khâu dọn cuối chưa gỡ được — soi tay."""
    from subtitles_extractor.application.services.line_echo_guard import (
        find_line_echoes,
    )

    return [
        {
            "index": echo.index,
            "neighbors": list(echo.neighbors),
            "text": translated_by_index.get(echo.index, ""),
        }
        for echo in find_line_echoes(source_by_index, translated_by_index)
    ]


def _address_rule_violations(
    source_by_index: dict[int, str],
    translated_by_index: dict[int, str],
    visual_cues: str,
    characters: str,
) -> list[dict[str, Any]]:
    """[v3.23.554] Dòng lệch quy tắc xưng hô của roster mà lưới tự động không sửa được.

    Phần sửa được đã do :mod:`address_rule_guard` sửa ngay trong lúc dịch, nên ở đây chỉ
    liệt kê phần còn lại (đại từ trùng danh từ thân tộc, vị trí sở hữu, hoặc cả cảnh đã
    đổi giọng) — người dùng tự quyết.
    """
    from subtitles_extractor.application.services.address_rule_guard import (
        find_address_rule_violations,
    )

    if not (visual_cues or "").strip() or not (characters or "").strip():
        return []
    return [
        {
            "index": item.index,
            "speaker": item.speaker,
            "addressee": item.addressee,
            "found": item.pronoun_found,
            "expected": item.pronoun_expected,
            "reason": item.reason,
        }
        for item in find_address_rule_violations(
            source_by_index, translated_by_index, visual_cues, characters
        )
        if not item.fixable
    ]


def detect_quality_flags(
    source_events: Iterable[_EventLike],
    translated_events: Iterable[_EventLike],
    glossary: str = "",
    characters: str = "",
    visual_cues: str = "",
) -> dict[str, Any]:
    """Phát hiện (heuristic) các dòng dịch KHẢ NGHI để soi nhanh chất lượng.

    Cảnh báo: đây là HEURISTIC — có thể dương tính giả (tên riêng/số/"OK" vốn giữ
    nguyên, hoặc tỉ lệ độ dài lệch do khác biệt ngôn ngữ). Chỉ để khoanh vùng.

    Args:
        source_events: Phụ đề gốc.
        translated_events: Phụ đề sau dịch.

    Returns:
        Dict gồm các danh sách chỉ số dòng khả nghi và bộ đếm tổng quan.
    """
    from subtitles_extractor.application.services.spoken_line_guard import (
        has_spoken_words,
        is_non_verbal_source,
    )

    source_by_index = {int(e.index): e.text for e in source_events}
    translated_list = list(translated_events)

    identical_to_source: list[int] = []
    empty_translation: list[int] = []
    unspoken_lines: list[int] = []
    length_anomaly: list[dict[str, Any]] = []

    for event in translated_list:
        idx = int(event.index)
        source_text = source_by_index.get(idx)
        if source_text is None:
            continue
        src = source_text.strip()
        dst_full = event.text.strip()
        if not src:
            continue
        if not dst_full:
            empty_translation.append(idx)
            continue
        # [v3.23.605] Câu gốc có lời mà bản dịch chỉ còn tag/chú thích ("[cười]"): phụ đề
        # xuất ra lọc mất tag, TTS bỏ trống dòng — mất dòng thoại mà cờ "rỗng" không thấy.
        if not has_spoken_words(dst_full) and not is_non_verbal_source(src):
            unspoken_lines.append(idx)
        # Bỏ tag người nói khỏi bản dịch trước khi đối chiếu/đo độ dài.
        dst = _strip_speaker_tag(dst_full)
        if dst and dst == src and not _is_untranslatable_source(src):
            identical_to_source.append(idx)
        # Cờ lệch độ dài: CJK-aware + bỏ qua nguồn quá ngắn (tránh dương tính giả).
        eff_src = _effective_source_length(src)
        if dst and eff_src >= _MIN_EFFECTIVE_LEN:
            ratio = len(dst) / eff_src
            # [v3.23.404] Ngưỡng CAO cận trên theo ĐỘ DÀI: nguồn CJK NGẮN (đặc biệt thành ngữ
            # 4 chữ 成语 như 赴汤蹈火, 危在旦夕) nở ra 2.5–3.3× là ĐÚNG (1 thành ngữ = cả cụm
            # mô tả). Dữ liệu thật: 15/15 cờ cũ đều là thành ngữ dịch chuẩn (dương tính giả).
            # Nguồn ngắn (eff < 18 ~ ≤6 ký tự CJK) dùng ngưỡng nới; nguồn dài giữ ngưỡng chặt.
            high = _LENGTH_RATIO_HIGH_SHORT if eff_src < _SHORT_SOURCE_EFF else _LENGTH_RATIO_HIGH
            if ratio < _LENGTH_RATIO_LOW or ratio > high:
                length_anomaly.append({"index": idx, "length_ratio": round(ratio, 2)})

    # [v3.23.465] Cờ DỊCH THIẾU (job-median): nhạy hơn length_anomaly vì so với TRUNG VỊ tỉ lệ
    # độ dài của CHÍNH phim (bắt được dòng lệch/sót mà ngưỡng tuyệt đối bỏ lọt — app_log: guard
    # bắt 52 dòng, length_anomaly chỉ 8). Đưa vào QC có cấu trúc để soi/sửa nhanh.
    under_translation: list[dict[str, Any]] = []
    try:
        from subtitles_extractor.application.services.under_translation_guard import (
            find_under_translated,
        )

        pairs = [
            (idx, source_by_index[idx].strip(), event.text.strip())
            for event in translated_list
            for idx in (int(event.index),)
            if source_by_index.get(idx, "").strip() and event.text.strip()
        ]
        under_translation = [
            {"index": s.index, "ratio": round(s.ratio, 2)}
            for s in find_under_translated(pairs)
        ]
    except (ImportError, ValueError, ZeroDivisionError):
        under_translation = []

    # [v3.23.139] DÒNG DỊCH TRÙNG LIỀN KỀ: dấu hiệu model bị "lệch-gộp ngữ nghĩa" — khi
    # một câu bị tách qua nhiều dòng, model dồn cả câu vào dòng đầu rồi KÉO nội dung dòng
    # sau lên, làm hai dòng liền kề có bản dịch TRÙNG nhau dù NGUỒN KHÁC nhau. Cờ này giúp
    # khoanh vùng lỗi căn dòng mà cờ đếm/độ dài không bắt được (số dòng vẫn khớp).
    duplicate_adjacent: list[dict[str, Any]] = []
    trans_sorted = sorted(
        ((int(e.index), e.text) for e in translated_list), key=lambda pair: pair[0]
    )
    for (idx1, text1), (idx2, text2) in pairwise(trans_sorted):
        dst1 = _strip_speaker_tag((text1 or "").strip())
        dst2 = _strip_speaker_tag((text2 or "").strip())
        if not dst1 or dst1 != dst2:
            continue
        # [v3.23.557] So theo NỘI DUNG: nguồn chỉ khác BẢNG MÃ dấu câu thì bản dịch
        # giống hệt là ĐÚNG, không phải lệch dòng. [v3.23.580] Cũng đúng khi chỉ khác
        # trợ từ ngữ khí — cùng quy tắc với khâu dọn dòng trùng.
        src1 = source_content_key(source_by_index.get(idx1) or "")
        src2 = source_content_key(source_by_index.get(idx2) or "")
        if src1 != src2:  # nguồn khác nhau mà dịch giống hệt -> nghi lệch dòng
            duplicate_adjacent.append({"indices": [idx1, idx2]})

    # [v3.23.590] Cặp dòng KHÔNG liền kề mang cùng bản dịch dù câu gốc khác hẳn: thường một
    # dòng mang lời của dòng kia (ca thật: "一二三" ra lời bài hát của dòng cách đó 24 dòng).
    from subtitles_extractor.application.services.distant_duplicate_guard import (
        find_distant_duplicates,
    )

    duplicate_distant = [
        {"indices": [first, second]}
        for first, second in find_distant_duplicates(
            {idx: text or "" for idx, text in source_by_index.items()},
            {idx: _strip_speaker_tag((text or "").strip()) for idx, text in trans_sorted},
        )
    ]

    # [v3.23.426] Cờ CJK còn sót: với đích KHÔNG phải CJK (vd Việt), chữ Hán/Nhật/Hàn trong
    # BẢN DỊCH nghĩa là còn chữ chưa dịch xong (vd '文人士大夫' → 'văn nhân sĩ đại夫' sót '夫').
    # Dữ liệu thật cho thấy đôi khi 1 chữ lọt qua cả 4 giai đoạn vào bản cuối → cần hiện lên QC.
    # Xác định đích có phải CJK không (heuristic): nếu ĐA SỐ ký tự bản dịch là CJK → đích là
    # CJK (vd dịch SANG tiếng Trung) → KHÔNG gắn cờ này.
    total_chars = 0
    total_cjk = 0
    for event in translated_list:
        text = event.text or ""
        total_chars += len(text)
        total_cjk += len(_CJK_RE.findall(text))
    _target_is_cjk = total_chars > 0 and (total_cjk / total_chars) > 0.5

    cjk_leftover: list[dict[str, Any]] = []
    if not _target_is_cjk:
        for event in translated_list:
            idx = int(event.index)
            residual = _CJK_RE.findall(event.text or "")
            if residual:
                cjk_leftover.append(
                    {"index": idx, "chars": "".join(dict.fromkeys(residual))}
                )

    # [v3.23.438] Cờ TỐC ĐỘ ĐỌC (CPS - ký tự/giây): phụ đề CJK→Việt thừa hưởng timing thiết kế
    # cho tiếng Trung cô đọng, mà tiếng Việt dài hơn → nhiều dòng đọc KHÔNG KỊP. Chỉ gắn cờ dòng
    # có thời lượng ĐÁNG TIN (≥ _READING_SPEED_MIN_DUR, loại timing OCR lỗi <0.8s) VÀ vượt ngưỡng
    # khó đọc (> _READING_SPEED_MAX_CPS) → khoanh vùng dòng dịch DÀI thật cần cô đọng/giãn giờ.
    reading_speed: list[dict[str, Any]] = []
    tts_compression: list[dict[str, Any]] = []
    for event in translated_list:
        text = (event.text or "").replace("\n", " ").strip()
        if not text:
            continue
        try:
            duration = float(event.end_sec) - float(event.start_sec)
        except (TypeError, ValueError, AttributeError):
            continue
        if duration < _READING_SPEED_MIN_DUR:
            continue
        cps = len(text) / duration
        if cps > _READING_SPEED_MAX_CPS:
            reading_speed.append(
                {"index": int(event.index), "cps": round(cps, 1),
                 "duration_sec": round(duration, 2), "chars": len(text)}
            )
        # [v3.23.444] Ước lượng NÉN audio TTS (mặt tiếng nói của cùng vấn đề). Chỉ xét dòng
        # timing đáng tin (đã lọc <_READING_SPEED_MIN_DUR ở trên) — dòng OCR timing lỗi sẽ
        # được v440 giãn giờ xử, không phải lỗi dịch.
        natural_tts_s = _TTS_DURATION_BASE_S + _TTS_DURATION_PER_CHAR_S * len(text)
        ratio = natural_tts_s / duration
        if ratio > _TTS_COMPRESSION_MAX_RATIO:
            tts_compression.append(
                {"index": int(event.index), "ratio": round(ratio, 2),
                 "duration_sec": round(duration, 2), "chars": len(text)}
            )

    # [v3.23.442] Cờ NHẤT QUÁN trong tập: dòng nguồn GIỐNG HỆT nhưng dịch KHÁC nhau (BỎ QUA
    # khác dấu câu/hoa-thường cuối câu — chỉ bắt khác biệt THẬT như 'Sô'/'Tô', đổi thứ tự từ).
    # Với phim bộ, nhất quán tên riêng/thuật ngữ rất quan trọng.
    source_to_indices: dict[str, list[int]] = {}
    for index, source_text in source_by_index.items():
        key = (source_text or "").strip()
        if len(key) >= 4 and _CJK_RE.search(key):
            source_to_indices.setdefault(key, []).append(index)

    translated_by_index = {int(e.index): (e.text or "") for e in translated_list}
    consistency: list[dict[str, Any]] = []
    for source_text, indices in source_to_indices.items():
        if len(indices) < 2:
            continue
        variants: dict[str, int] = {}
        for idx in indices:
            normalized = _normalize_for_consistency(translated_by_index.get(idx, ""))
            if normalized:
                variants.setdefault(normalized, idx)
        if len(variants) > 1:
            consistency.append({
                "source": source_text[:30],
                "indices": sorted(indices),
                "variants": [translated_by_index.get(i, "") for i in variants.values()],
            })

    shifted_lines = _shifted_lines(source_by_index, translated_list, glossary, characters)
    misaligned_blocks = _misaligned_blocks(source_by_index, translated_list)
    # [v3.23.562] Số liệu xưng hô theo cặp chỉ tính cue có CẢ HAI phía là nhân vật trong
    # bảng — nhãn chung chung ("Nữ chính") gom nhiều người vào một cặp, sinh cờ ảo.
    from subtitles_extractor.application.services.cue_label_canonicalizer import (
        roster_dialogue_cues_json,
    )

    pair_cues = roster_dialogue_cues_json(visual_cues, characters)
    # [v3.23.577] Nhãn "người nói = người nghe" (độc thoại) không phải một CẶP xưng hô —
    # đo riêng ở ``monologue_self_forms``; độ phủ cues vẫn tính cả chúng như trước.
    dialogue_cues = _without_self_address(pair_cues)
    address_flags, address_pairs_examined = address_pair_stats(
        dialogue_cues, translated_list
    )
    cue_coverage = address_cue_coverage(pair_cues, set(translated_by_index))
    drift_flags = address_drift(dialogue_cues, source_by_index, translated_list)
    monologue_flags = _monologue_self_forms(
        visual_cues, source_by_index, translated_by_index, characters
    )
    number_issues = _number_issues(source_by_index, translated_by_index)
    outdated_rules = _outdated_address_rules(
        source_by_index, translated_by_index, visual_cues, characters
    )
    rule_violations = _address_rule_violations(
        source_by_index, translated_by_index, visual_cues, characters
    )
    echo_lines = _echo_lines(source_by_index, translated_by_index)
    from subtitles_extractor.application.services.roster_format import address_clauses

    address_rule_count = len(address_clauses(characters))
    glossary_violations, suspect_glossary_entries = split_suspect_entries(
        _glossary_violations(source_by_index, translated_list, glossary)
    )
    return {
        "note": (
            "Heuristic — length_anomaly đã CJK-aware (quy đổi độ dài nguồn CJK) và "
            "bỏ tag người nói; vẫn có thể dương tính giả (tên riêng/số giữ nguyên). "
            "duplicate_adjacent: 2 dòng liền kề dịch trùng dù nguồn khác. "
            "duplicate_distant: 2 dòng KHÔNG liền kề (cách nhau tới 60 dòng) dịch giống "
            "hệt dù nguồn khác hẳn — một dòng mang lời của dòng kia, hoặc hai câu đồng "
            "nghĩa; khâu dịch đã dịch lại các cặp này và chỉ thay dòng ra khác hẳn (nhật "
            "ký 'distant_duplicates_retranslated'), cặp còn lại ở đây cần soi tay. "
            "monologue_self_forms: dòng ĐỘC THOẠI (nhãn cues người nói = người nghe) mà "
            "bản dịch tự xưng bằng từ vai vế (em/anh/chị…) như đang nói với ai — "
            "lines là các dòng đó, variants là mọi cách tự xưng khi độc thoại của người "
            "đó. "
            "echo_lines: dòng dịch CHỨA gần trọn câu của dòng kề (neighbors) dù câu gốc "
            "không lặp — model gộp câu rồi vẫn dịch dòng kề, giọng đọc nói một ý hai "
            "lần. "
            "cjk_leftover: bản dịch còn chữ CJK chưa dịch (khi đích không phải CJK). "
            "unspoken_lines: câu gốc có lời nhưng bản dịch chỉ còn tag/chú thích trong "
            "ngoặc (vd '[cười]') — phụ đề lọc mất tag, lồng tiếng bỏ trống dòng. "
            "reading_speed: dòng đọc không kịp (CPS cao) dù thời lượng đáng tin. "
            "consistency: nguồn giống hệt nhưng dịch khác (bỏ qua khác dấu câu). "
            "tts_compression: dòng buộc TTS nén audio mạnh (>1.6x) → giọng chói/méo. "
            "misaligned_blocks: KHỐI dòng mà tương quan độ dài nguồn/dịch sụp đổ — dấu hiệu "
            "model gộp hoặc bỏ dòng rồi đẩy lệch cả đoạn; đây là lỗi NẶNG nhất, soi trước. "
            "final_vs_literal_shifts: dải dòng bản cuối mang câu của dòng lệch so "
            "với bản dịch thô (offset = lệch bao nhiêu dòng) — giai đoạn tinh chỉnh đẩy "
            "lệch, hoặc bản thô lệch và đã được sửa; soi tay cùng nhật ký 'stage_shift'. "
            "final_vs_literal_rewrites: dải dòng bản cuối bị VIẾT LẠI khác hẳn bản dịch "
            "thô (không giống cả dòng kề) — giai đoạn tinh chỉnh bịa lời, hoặc bản thô "
            "lệch và đã được sửa; soi tay cùng nhật ký 'stage_rewrite'. "
            "address_rule_violations: dòng xưng hô khác quy tắc bảng nhân vật mà lưới tự "
            "động KHÔNG dám sửa (kèm lý do) — cần soi tay. "
            "address_rule_outdated: quy tắc xưng hô trong bảng nhân vật mà CẢ PHIM dùng "
            "khác — nên sửa bảng nhân vật rồi phân tích lại. "
            "number_issues: con số dịch SAI GIÁ TRỊ (mismatch), lệch lối đọc của cả "
            "phim — 'vạn' trong phim đọc nghìn/triệu (non_standard_unit) hay "
            "nghìn/triệu trong phim đọc 'vạn' (standard_unit) — hoặc đúng giá trị mà "
            "sai cách đọc như 'một mươi triệu' (malformed). "
            "counts.address_cue_coverage: tỉ lệ dòng có Visual Cues đủ cặp người nói → "
            "người nghe (cả hai là nhân vật trong bảng, nếu có bảng) — MẪU SỐ của mọi số "
            "liệu xưng hô; dưới ~0.6 thì số liệu xưng hô 'đẹp' chỉ vì lưới không nhìn "
            "thấy phần lớn phim. "
            "counts.address_rules: số mệnh đề xưng hô đọc được trong bảng nhân vật — "
            "bằng 0 với bảng nhiều nhân vật thì mọi address_rule_* = 0 KHÔNG có nghĩa là "
            "tốt (không có quy tắc để so; thường do dùng lại phân tích từ bản cũ) → nên "
            "Phân tích lại."
        ),
        "identical_to_source_indices": identical_to_source,
        "empty_translation_indices": empty_translation,
        "unspoken_lines": unspoken_lines,
        "length_anomaly": length_anomaly,
        "duplicate_adjacent": duplicate_adjacent,
        "duplicate_distant": duplicate_distant,
        # [v3.23.576] Nói lặp dòng kề — phần khâu dọn cuối chưa gỡ được.
        "echo_lines": echo_lines[:30],
        "cjk_leftover": cjk_leftover,
        "under_translation": under_translation,
        "reading_speed": reading_speed,
        "consistency": consistency,
        "glossary_violations": glossary_violations,
        "suspect_glossary_entries": suspect_glossary_entries,
        "shifted_lines": shifted_lines[:50],
        # [v3.23.549] Lệch theo KHỐI — phủ được cả vùng không có mốc neo nào (mốc neo chỉ phủ ~9%
        # số dòng). Ca thật: khối 18 dòng sai toàn bộ mà mọi cờ khác đều im lặng.
        "misaligned_blocks": misaligned_blocks[:20],
        # [v3.23.524] Xưng hô lộn xộn theo cặp nhân vật — tiếng Việt không có đại từ trung tính
        # nên đây là lỗi người xem nhận ra ngay, mà không cờ nào khác bắt được.
        "address_inconsistency": address_flags[:30],
        # [v3.23.551] Xưng hô đổi TRONG CÙNG CẢNH, tách theo VAI (tự xưng / gọi đối phương) nhờ
        # 我/你 ở câu gốc — chính xác hơn hẳn cờ gộp cả phim ở trên.
        "address_drift": drift_flags[:40],
        # [v3.23.577] Độc thoại tự xưng như đang nói với ai — không phải một cặp xưng hô.
        "monologue_self_forms": monologue_flags[:20],
        # [v3.23.554] Lệch quy tắc xưng hô ĐÃ GHI RÕ trong bảng nhân vật. Phần máy tự sửa
        # được đã được sửa trong lúc dịch; danh sách này là phần còn lại, cần soi tay.
        "address_rule_violations": rule_violations[:40],
        # [v3.23.557] Quy tắc xưng hô trong BẢNG NHÂN VẬT mà cả phim dùng khác — sửa một
        # dòng roster là xong, không phải soi tay từng dòng phụ đề.
        "address_rule_outdated": outdated_rules[:20],
        # [v3.23.556] Số tiền/số lượng dịch sai giá trị — câu vẫn trôi chảy nên mọi cờ
        # khác đều im lặng, nhưng người xem nhận ra ngay.
        "number_issues": number_issues[:30],
        "tts_compression": tts_compression,
        "counts": {
            "identical_to_source": len(identical_to_source),
            "empty_translation": len(empty_translation),
            "unspoken_lines": len(unspoken_lines),
            "length_anomaly": len(length_anomaly),
            "duplicate_adjacent": len(duplicate_adjacent),
            "duplicate_distant": len(duplicate_distant),
            "echo_lines": len(echo_lines),
            "cjk_leftover": len(cjk_leftover),
            "under_translation": len(under_translation),
            "reading_speed": len(reading_speed),
            "consistency": len(consistency),
            # [v3.23.550] Số chính chỉ đếm TÊN RIÊNG — loại bắt buộc sửa. Biểu đạt đếm riêng.
            "glossary_violations": sum(
                v["violations"] for v in glossary_violations
                if v.get("kind") == GLOSSARY_KIND_NAME
            ),
            "glossary_expression_deviations": sum(
                v["violations"] for v in glossary_violations
                if v.get("kind") != GLOSSARY_KIND_NAME
            ),
            "suspect_glossary_entries": len(suspect_glossary_entries),
            "shifted_lines": len(shifted_lines),
            "misaligned_blocks": len(misaligned_blocks),
            "misaligned_block_lines": sum(int(b["lines"]) for b in misaligned_blocks),
            "address_inconsistency": len(address_flags),
            # Mẫu số để so sánh CÔNG BẰNG giữa các lần chạy (độ phủ cues dao động mạnh).
            "address_pairs_examined": address_pairs_examined,
            # [v3.23.559] Mẫu số theo DÒNG: ca thật cues chỉ phủ 27% → số cặp xét giảm
            # 67 → 23 và mọi cờ xưng hô giảm theo, trông như chất lượng tốt lên.
            "address_cue_coverage": cue_coverage,
            "address_drift": len(drift_flags),
            # [v3.23.578] Mẫu số của address_rule_*: bảng không có quy tắc thì các cờ
            # đó bằng 0 vì không có gì để so, không phải vì dịch đúng.
            "address_rules": address_rule_count,
            "monologue_kin_self": sum(len(item["lines"]) for item in monologue_flags),
            "address_rule_violations": len(rule_violations),
            "address_rule_outdated": len(outdated_rules),
            "number_issues": len(number_issues),
            "tts_compression": len(tts_compression),
        },
    }


def _event_to_dict(event: _EventLike) -> dict[str, Any]:
    confidence = getattr(event, "confidence", None)
    conf_value: float | None
    try:
        conf_value = round(float(confidence), 4) if confidence is not None else None
    except (TypeError, ValueError):
        conf_value = None
    return {
        "index": int(event.index),
        "start_sec": round(float(event.start_sec), 3),
        "end_sec": round(float(event.end_sec), 3),
        "text": event.text,
        "confidence": conf_value,
    }


def _literal_and_final(
    stages: list[dict[str, Any]], translated_list: list[dict[str, Any]]
) -> tuple[dict[int, str], dict[int, str]] | None:
    """``index → câu`` của bản dịch thô và bản cuối; None nếu không có bản thô."""
    literal = next((s for s in stages if s.get("stage_id") == "literal"), None)
    lines = literal.get("lines") if literal else None
    if not isinstance(lines, list) or not lines:
        return None
    previous = {
        int(line["index"]): str(line.get("text", "") or "")
        for line in lines
        if isinstance(line, dict) and str(line.get("index", "")).isdigit()
    }
    current = {int(e["index"]): str(e.get("text", "") or "") for e in translated_list}
    return previous, current


def _shift_rows(shifts: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "from_index": shift.from_index,
            "to_index": shift.to_index,
            "offset": shift.offset,
            "lines": shift.lines,
        }
        for shift in shifts
    ]


def final_vs_literal_shifts(
    stages: list[dict[str, Any]], translated_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """[v3.23.569] Dải dòng bản CUỐI mang câu của dòng lệch so với bản dịch THÔ (thuần).

    Bản dịch thô là bản duy nhất dịch thẳng từ câu gốc; bản cuối lệch so với nó theo cả
    một dải là dấu hiệu giai đoạn tinh chỉnh đã đẩy lệch dòng (hoặc chính bản thô lệch và
    đã được sửa) — vùng cần soi tay đầu tiên.

    Args:
        stages: Các giai đoạn đã lưu (``stage_id`` + ``lines``).
        translated_list: Bản dịch cuối (``index`` + ``text``).

    Returns:
        ``[{"from_index", "to_index", "offset", "lines"}]``; rỗng nếu thiếu bản thô.
    """
    from subtitles_extractor.application.services.stage_shift_guard import (
        find_stage_shifts,
    )

    pair = _literal_and_final(stages, translated_list)
    return _shift_rows(find_stage_shifts(*pair)) if pair else []


def final_vs_literal_rewrites(
    stages: list[dict[str, Any]], translated_list: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """[v3.23.589] Dải dòng bản CUỐI bị viết lại khác hẳn bản dịch THÔ (hàm thuần).

    Khác dải lệch: lời mới không giống cả dòng kề của bản thô. Ca thật v3.23.588, dòng
    68-77: bản thô "Đừng giận.", bản cuối "Khai mau lên!" — giai đoạn tinh chỉnh bịa lời
    theo tiếng đám đông. Khâu dịch đã phân xử các dải này bằng trọng tài (nhật ký
    ``stage_rewrite``); dải còn lại ở đây là dải trọng tài giữ nguyên hoặc không phân
    định được, hoặc dải do hai giai đoạn cộng lại — cần soi tay.

    Args:
        stages: Các giai đoạn đã lưu (``stage_id`` + ``lines``).
        translated_list: Bản dịch cuối (``index`` + ``text``).

    Returns:
        ``[{"from_index", "to_index", "offset", "lines"}]`` (``offset`` luôn 0); rỗng
        nếu thiếu bản thô.
    """
    from subtitles_extractor.application.services.stage_shift_guard import (
        find_stage_rewrites,
        find_stage_shifts,
    )

    pair = _literal_and_final(stages, translated_list)
    if pair is None:
        return []
    return _shift_rows(find_stage_rewrites(*pair, find_stage_shifts(*pair)))


def _attach_final_vs_literal_shifts(
    quality_flags: dict[str, Any],
    stages: list[dict[str, Any]],
    translated_list: list[dict[str, Any]],
) -> None:
    shifts = final_vs_literal_shifts(stages, translated_list)
    rewrites = final_vs_literal_rewrites(stages, translated_list)
    quality_flags["final_vs_literal_shifts"] = shifts[:20]
    quality_flags["final_vs_literal_rewrites"] = rewrites[:20]
    counts = quality_flags.get("counts")
    if isinstance(counts, dict):
        counts["final_vs_literal_shifts"] = len(shifts)
        counts["final_vs_literal_rewrites"] = len(rewrites)


def _parse_stage_lines(lines_json: str) -> Any:
    """Giải mã ``lines_json`` (chuỗi JSON) thành đối tượng; lỗi → trả raw + cờ lỗi."""
    if not lines_json:
        return []
    try:
        parsed = json.loads(lines_json)
    except (json.JSONDecodeError, TypeError):
        return {"_parse_error": True, "raw": lines_json}
    # [v3.23.609] ``h`` là vân tay câu gốc để kiểm khớp khi mở lại phiên — câu gốc đã có
    # đủ trong ``source_events`` nên bỏ khỏi file chuẩn đoán cho gọn.
    if isinstance(parsed, list):
        return [
            {key: value for key, value in row.items() if key != "h"}
            if isinstance(row, dict) else row
            for row in parsed
        ]
    return parsed


#: [v3.23.579] Ghi chú khi nhật ký trong bộ nhớ thuộc lần dịch khác.
FOREIGN_PROCESS_LOG_NOTE = (
    "Nhật ký diễn biến trong bộ nhớ thuộc lần dịch KHÁC (phim khác, hoặc dịch lại một "
    "số dòng) nên không đính kèm để khỏi gán nhầm. Muốn có nhật ký, hãy xuất chuẩn đoán "
    "ngay sau khi dịch phim này."
)


def _process_log_snapshot(source_texts: list[str]) -> dict[str, Any]:
    """[v3.23.519] Diễn biến quá trình dịch; rỗng nếu không có (không bao giờ ném lỗi).

    [v3.23.579] Chỉ đính kèm khi nhật ký thuộc ĐÚNG phụ đề đang xuất (so dấu vân tay).

    Args:
        source_texts: Văn bản phụ đề nguồn đang xuất.
    """
    try:
        from subtitles_extractor.application.services.translation_process_log import (
            get_process_log,
            run_fingerprint,
        )

        snapshot = get_process_log().snapshot()
        owner = str(snapshot.get("owner", "") or "")
        if owner and owner != run_fingerprint(source_texts):
            return {
                "events": [],
                "counts": {},
                "dropped": 0,
                "owner": owner,
                "foreign_run": True,
                "note": FOREIGN_PROCESS_LOG_NOTE,
            }
        return snapshot
    except Exception:  # noqa: BLE001 — thiếu nhật ký không được cản việc xuất chuẩn đoán
        return {"events": [], "counts": {}, "dropped": 0}


def build_diagnostics_bundle(
    *,
    app_version: str,
    exported_at: str,
    video_path: str,
    source_events: Iterable[_EventLike],
    translated_events: Iterable[_EventLike],
    session: Any | None,
    series_context: Any | None = None,
) -> dict[str, Any]:
    """Gom toàn bộ dữ liệu một lần dịch thành dict JSON-serializable.

    Args:
        app_version: Phiên bản ứng dụng (để biết bản nào sinh dữ liệu).
        exported_at: Mốc thời gian xuất (ISO-8601).
        video_path: Đường dẫn video nguồn (str).
        source_events: Phụ đề GỐC (trước dịch).
        translated_events: Phụ đề KẾT QUẢ (sau dịch, bản hiện hành).
        session: ``TranslationSession`` (phân tích + giai đoạn) hoặc ``None``.
        series_context: Ngữ cảnh phim bộ (glossary/roster/tóm tắt chung) hoặc ``None``.

    Returns:
        Dict mô tả đầy đủ quá trình dịch, an toàn để ``json.dump``.
    """
    from pathlib import PurePath

    source_events = list(source_events)
    translated_events = list(translated_events)
    source_list = [_event_to_dict(e) for e in source_events]
    translated_list = [_event_to_dict(e) for e in translated_events]
    # [v3.23.500] Truyền glossary để đo mức TUÂN THỦ bảng thuật ngữ trong bản dịch cuối.
    _session_glossary = str(getattr(session, "analysis_glossary", "") or "")
    quality_flags = detect_quality_flags(
        source_events,
        translated_events,
        glossary=_session_glossary,
        characters=str(getattr(session, "analysis_characters", "") or ""),
        visual_cues=str(getattr(session, "analysis_visual_cues", "") or ""),
    )

    analysis: dict[str, Any] = {}
    stages: list[dict[str, Any]] = []
    cloud_files: list[dict[str, Any]] = []
    if session is not None:
        analysis = {
            "source_lang": getattr(session, "analysis_source_lang", ""),
            "target_lang": getattr(session, "analysis_target_lang", ""),
            "characters": getattr(session, "analysis_characters", ""),
            "overview": getattr(session, "analysis_overview", ""),
            "glossary": getattr(session, "analysis_glossary", ""),
            "visual_cues": getattr(session, "analysis_visual_cues", ""),
            "input_hash": getattr(session, "analysis_input_hash", ""),
        }
        for stage in getattr(session, "stages", ()) or ():
            stages.append(
                {
                    "stage_id": getattr(stage, "stage_id", ""),
                    "input_hash": getattr(stage, "input_hash", ""),
                    "completed_at": getattr(stage, "completed_at", ""),
                    "lines": _parse_stage_lines(getattr(stage, "lines_json", "")),
                }
            )
        for cloud in getattr(session, "cloud_files", ()) or ():
            cloud_files.append(
                {
                    "remote_name": getattr(cloud, "remote_name", ""),
                    "start_sec": getattr(cloud, "start_sec", 0.0),
                    "end_sec": getattr(cloud, "end_sec", 0.0),
                    "uploaded_at": getattr(cloud, "uploaded_at", ""),
                }
            )

    _attach_final_vs_literal_shifts(quality_flags, stages, translated_list)

    series: dict[str, Any] = {}
    if series_context is not None:
        series = {
            "glossary": getattr(series_context, "glossary", ""),
            "characters": getattr(series_context, "characters", ""),
            "overview": getattr(series_context, "overview", ""),
        }

    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "app_version": app_version,
        "exported_at": exported_at,
        "video": {
            "path": video_path,
            "filename": PurePath(video_path).name if video_path else "",
        },
        "languages": {
            "source": analysis.get("source_lang", ""),
            "target": analysis.get("target_lang", ""),
        },
        "counts": {
            "source_events": len(source_list),
            "translated_events": len(translated_list),
            "stages": len(stages),
            "cloud_files": len(cloud_files),
        },
        "analysis": analysis,
        "series_context": series,
        "quality_flags": quality_flags,
        "cloud_files": cloud_files,
        "source_events": source_list,
        "final_translation": translated_list,
        "stages": stages,
        # [v3.23.519] Diễn biến quá trình dịch: lô nào bị model trả sai và thử lại mấy lần, lô nào
        # bị phát hiện LỆCH DÒNG, lô nào đành giữ nguyên văn bản gốc. Trước đây các thông tin này
        # chỉ nằm trong nhật ký ứng dụng và mất khi đóng chương trình, nên mỗi lần chuẩn đoán lại
        # phải xin thêm app_log và không thể đối chiếu "lô bị thử lại" với "dòng dịch kém".
        "process_log": _process_log_snapshot(
            [str(getattr(event, "text", "") or "") for event in source_events]
        ),
    }


VIETNAMESE_PRONOUNS = (
    "tôi", "ta", "tao", "tớ", "anh", "em", "chị", "cô", "cậu", "mày",
    "ông", "bà", "chú", "bác", "con", "cháu", "ngươi", "thiếp", "chàng", "nàng",
)
"""Đại từ xưng hô tiếng Việt cần theo dõi tính nhất quán."""

MIN_PAIR_MENTIONS = 6
"""Cặp nhân vật phải có đủ ngần này lượt xưng hô mới đủ căn cứ kết luận."""

KIN_COMPOUND_TAIL = (
    "trai|gái|dâu|rể|hai|ba|cả|út|họ|chồng|vợ|ruột|nuôi|kết nghĩa|sĩ|giáo|chủ|nhóc|"
    "bé|chàng|nàng"
)
"""[v3.23.583] Từ đứng sau biến từ thân tộc thành DANH TỪ: "bác sĩ", "chị dâu", "anh
trai", "ông chủ", "cô giáo", "cậu nhóc". Ca thật v3.23.582: cặp ``林逸 → 主治医生`` bị báo
xưng hô lộn xộn với biến thể 'bác' (4 lượt — đều là "Bác sĩ!") và 'chị' (3 lượt — đều là
"chị dâu tôi"), trong khi người được gọi là ông bác sĩ."""

_PURE_PRONOUNS = frozenset({"tôi", "tao", "tớ", "ta", "mình"})
_KIN_COMPOUND_AFTER_RE = re.compile(rf"(?i)\s+(?:{KIN_COMPOUND_TAIL})(?![\w])")


def is_kin_compound_at(text: str, match: re.Match[str]) -> bool:
    """Từ thân tộc tại ``match`` là một phần danh từ ghép ("bác sĩ"), không phải đại từ.

    Đại từ thuần (tôi/tao/tớ/ta/mình) không bao giờ ghép: "Đều tại tớ cả" = "tại tớ hết".

    Args:
        text: Câu dịch.
        match: Vị trí một từ xưng hô trong ``text`` (nhóm 0 là chính từ đó).

    Returns:
        True nếu ngay sau từ đó là hậu tố tạo danh từ.
    """
    if match.group(0).lower() in _PURE_PRONOUNS:
        return False
    return _KIN_COMPOUND_AFTER_RE.match(text, match.end()) is not None


def address_words(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Các từ xưng hô (viết thường) ``pattern`` bắt được, bỏ từ nằm trong danh từ ghép."""
    return [
        match.group(0).lower()
        for match in pattern.finditer(text or "")
        if not is_kin_compound_at(text, match)
    ]

MIN_VARIANT_SHARE = 0.15
"""Một cách xưng hô chiếm từ tỉ lệ này trở lên mới tính là 'biến thể đang dùng'."""

MAX_CONSISTENT_VARIANTS = 3
"""Số kiểu xưng hô tối đa coi là còn nhất quán cho một cặp nhân vật.

[v3.23.536] Nâng 2 → 3 sau khi soi dữ liệu thật. Tiếng Việt luôn dùng MỘT CẶP đại từ (tự xưng +
gọi đối phương), nên riêng cách nói nhất quán đã chiếm 2 kiểu; thêm một biến thể ngữ cảnh hợp lệ là
thành 3. Ví dụ ``甄劍聲 → 洪震`` dùng "tao/mày" kèm đôi lần "tôi" — đó là giữ vai, không phải lộn
xộn, nhưng ngưỡng 2 vẫn gắn cờ. Ngưỡng 3 chỉ còn bắt các cặp trộn lẫn THẬT (vd vừa "cậu" vừa "anh"
vừa "em" cho cùng một người)."""


def address_pair_stats(
    visual_cues: str, translated_events: list[_EventLike]
) -> tuple[list[dict[str, Any]], int]:
    """[v3.23.525] Như :func:`address_inconsistencies` nhưng trả kèm SỐ CẶP ĐO ĐƯỢC.

    Vì sao cần: cờ này chỉ đo được ở các cặp có đủ dữ liệu trong Visual Cues, mà độ phủ cues dao
    động rất mạnh giữa các lần phân tích (đo thật: 58,5% xuống 16,9% khi đổi model phân tích). Khi
    đó số cặp đo được rơi từ 49 xuống 16, nên so sánh con số THÔ giữa hai lần chạy sẽ hiểu sai:
    "12 cặp lộn xộn" và "2 cặp lộn xộn" thực ra là 24,5% và 12,5%. Trả kèm mẫu số để đọc đúng.

    Args:
        visual_cues: Chuỗi JSON Visual Cues.
        translated_events: Các dòng đã dịch.

    Returns:
        Cặp ``(danh sách cặp lộn xộn, số cặp đủ dữ liệu để xét)``.
    """
    findings = address_inconsistencies(visual_cues, translated_events)
    examined = _count_examined_pairs(visual_cues, translated_events)
    return findings, examined


def _pair_address_usage(
    visual_cues: str, translated_events: list[_EventLike]
) -> dict[tuple[str, str], dict[str, int]]:
    """Số lượt dùng từng từ xưng hô theo cặp ``(người nói, người nghe)``.

    [v3.23.586] Bỏ ba thứ không phải xưng hô GIỮA hai người của cặp — trước đây bị đếm:

    * ngôi ba: "cô ấy", "ông ấy", "anh ta". Ca thật v3.23.585: cặp ``醫生 → 龍傳人`` bị
      báo lộn xộn với biến thể 'cô' (4 lượt, đều là "cô ấy" — bệnh nhân) và 'ông' (2/4
      lượt là "ông ấy" — trấn trưởng);
    * số nhiều: "chúng ta", "bọn tao", "tụi mày";
    * độc thoại: nhãn người nói trùng người nghe.

    Args:
        visual_cues: Chuỗi JSON Visual Cues (có ``id``/``spk``/``to``).
        translated_events: Các dòng đã dịch.

    Returns:
        ``{(người nói, người nghe): {từ xưng hô: số lượt}}``; rỗng khi không có cues.
    """
    texts = {int(event.index): (event.text or "") for event in translated_events}
    usage: dict[tuple[str, str], dict[str, int]] = {}
    for pair, line_nos in cue_lines_by_pair(visual_cues, set(texts)).items():
        counter = usage.setdefault(pair, {})
        for line_no in line_nos:
            for word in address_words(texts[line_no], ADDRESS_WORD_RE):
                counter[word] = counter.get(word, 0) + 1
    return usage


def _count_examined_pairs(visual_cues: str, translated_events: list[_EventLike]) -> int:
    """Đếm số cặp nhân vật có đủ lượt xưng hô để kết luận."""
    usage = _pair_address_usage(visual_cues, translated_events)
    totals = (sum(counter.values()) for counter in usage.values())
    return sum(1 for total in totals if total >= MIN_PAIR_MENTIONS)


def address_inconsistencies(
    visual_cues: str, translated_events: list[_EventLike]
) -> list[dict[str, Any]]:
    """[v3.23.524] Đo tính NHẤT QUÁN của xưng hô theo từng cặp nhân vật.

    Vì sao cần: tiếng Việt không có đại từ trung tính như "you" — mỗi cặp quan hệ có một cặp xưng
    hô riêng (anh/em, chị/cậu, tao/mày…). Model dịch theo từng lô nên dễ đổi cách gọi giữa chừng.
    Đo trên phim thật: cặp ``春花 → 宋钱`` dùng lẫn lộn "chị" 16 lượt, "em" 15, "cậu" 12 — người xem
    sẽ thấy quan hệ nhân vật nhảy loạn. Không cờ nào hiện có bắt được lỗi này.

    Dùng nhãn người nói/người nghe trong Visual Cues làm mốc, nên chỉ đo được ở phim có phân tích
    hình ảnh; không có cues thì trả rỗng.

    Args:
        visual_cues: Chuỗi JSON Visual Cues (có ``id``/``spk``/``to``).
        translated_events: Các dòng đã dịch.

    Returns:
        Danh sách cặp nhân vật xưng hô lộn xộn, sắp theo mức nghiêm trọng.
    """
    findings: list[dict[str, Any]] = []
    for (speaker, addressee), counter in _pair_address_usage(
        visual_cues, translated_events
    ).items():
        total = sum(counter.values())
        if total < MIN_PAIR_MENTIONS:
            continue
        variants = {
            word: count
            for word, count in counter.items()
            if count >= 2 and count / total >= MIN_VARIANT_SHARE
        }
        if len(variants) > MAX_CONSISTENT_VARIANTS:
            findings.append({
                "speaker": speaker,
                "addressee": addressee,
                "mentions": total,
                "variants": dict(sorted(variants.items(), key=lambda kv: -kv[1])),
            })
    findings.sort(key=lambda item: (-len(item["variants"]), -item["mentions"]))
    return findings


ADDRESS_SCENE_GAP_LINES = 40
"""Hai lượt thoại của cùng một cặp cách nhau quá ngần này dòng thì coi là KHÁC CẢNH."""

ADDRESS_DRIFT_MIN_LINES = 4
"""Một vai trong một cảnh phải có đủ ngần này dòng mới đủ căn cứ kết luận."""

ADDRESS_DRIFT_MIN_EACH = 2
"""Mỗi biến thể phải xuất hiện ít nhất ngần này lần — một lần lẻ là nhiễu."""

ADDRESS_DRIFT_MIN_SHARE = 0.25
"""Biến thể phải chiếm từ tỉ lệ này trở lên trong vai đó ở cảnh đó."""

SELF_SOURCE_RE = re.compile(r"[我俺咱](?![们們])")
YOU_SOURCE_RE = re.compile(r"[你您](?![们們])")
# [v3.23.554] Ngoài hậu tố 们/們, tiếng Trung còn dùng 俩/倆 và 两人 để chỉ SỐ NHIỀU.
# Ca thật v3.23.549: '谁让咱俩是一个村的' → "hai ta cũng là người một làng" từng bị
# coi là ngôi nhất SỐ ÍT, suýt bị lưới xưng hô đổi thành "hai anh".
PLURAL_SOURCE_RE = re.compile(r"[我你咱您俺](?:[们們俩倆]|[两兩]人)")
ADDRESS_WORD_RE = re.compile(
    r"(?<!chúng )(?<!bọn )(?<!tụi )(?<![\w])(" + "|".join(VIETNAMESE_PRONOUNS) + r")(?![\w])"
    r"(?!\s+(?:ấy|ta)\b)",
    re.IGNORECASE,
)


def cue_lines_by_pair(visual_cues: str, valid: set[int]) -> dict[tuple[str, str], list[int]]:
    """Gom chỉ số dòng theo cặp ``(người nói, người nghe)`` từ Visual Cues."""
    try:
        parsed = json.loads(visual_cues) if (visual_cues or "").strip() else []
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}
    if isinstance(parsed, dict):
        parsed = parsed.get("cues", [])
    if not isinstance(parsed, list):
        return {}
    by_pair: dict[tuple[str, str], list[int]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("spk") or "").strip()
        addressee = str(item.get("to") or "").strip()
        try:
            line_no = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        # [v3.23.557] Người nói TRÙNG người nghe là ĐỘC THOẠI (109 dòng ở phim thật) —
        # không có đối phương thì không có "xưng hô với ai", nên mọi phép đo xưng hô phải
        # bỏ qua. Trước đây sinh cờ ảo: cặp 宋钱→宋钱 "trôi" giữa 'tôi' và 'em'.
        if speaker and addressee and speaker != addressee and line_no in valid:
            by_pair.setdefault((speaker, addressee), []).append(line_no)
    return by_pair


def address_cue_coverage(visual_cues: str, valid: set[int]) -> float:
    """[v3.23.559] Tỉ lệ dòng đo được xưng hô (có cặp người nói → người nghe khác nhau).

    Args:
        visual_cues: Chuỗi JSON Visual Cues (``id``/``spk``/``to``).
        valid: Tập chỉ số dòng của bản dịch.

    Returns:
        Tỉ lệ trong [0, 1], làm tròn 4 chữ số; 0.0 khi không có dòng nào.
    """
    if not valid:
        return 0.0
    covered = {
        line_no
        for line_nos in cue_lines_by_pair(visual_cues, valid).values()
        for line_no in line_nos
    }
    return round(len(covered) / len(valid), 4)


def split_scenes(indices: list[int], gap: int = ADDRESS_SCENE_GAP_LINES) -> list[list[int]]:
    """Tách danh sách dòng (đã sắp) thành các CẢNH theo khoảng cách giữa hai lượt liên tiếp."""
    scenes: list[list[int]] = []
    for index in sorted(indices):
        if scenes and index - scenes[-1][-1] <= gap:
            scenes[-1].append(index)
        else:
            scenes.append([index])
    return scenes


def address_drift(
    visual_cues: str,
    source_by_index: dict[int, str],
    translated_events: Iterable[_EventLike],
) -> list[dict[str, Any]]:
    """[v3.23.551] Xưng hô ĐỔI GIỮA CHỪNG trong cùng một cảnh, tách theo VAI.

    Vì sao thay cách đo cũ: :func:`address_inconsistencies` gộp mọi đại từ của một cặp trên CẢ
    PHIM, nên (a) đổi giọng hợp lý giữa các cảnh (sàm sỡ ngoài suối "tao/mày" → nửa đêm ngọt nhạt
    "anh/em") bị báo lỗi, còn (b) trôi thật trong cùng cảnh ("cứu lấy **tôi**" → 5 dòng sau "nên
    **em** đến tìm anh") lại lọt vì "tôi" và "em" cùng bị đếm chung với đại từ gọi đối phương.

    Cách đo mới, tất định:

    * **Vai** lấy từ câu GỐC: dòng chỉ có ``我`` (không ``你``) thì đại từ tiếng Việt là TỰ XƯNG;
      chỉ có ``你/您`` thì là GỌI ĐỐI PHƯƠNG. Dòng có cả hai, hoặc số nhiều (``我们``), bỏ qua.
    * **Cảnh**: các lượt thoại của cặp cách nhau không quá :data:`ADDRESS_SCENE_GAP_LINES` dòng.
    * Bỏ ngôi ba ("cậu **ấy**", "anh **ta**") và ngôi nhất số nhiều ("**chúng** ta").

    Đo trên 3 lần chạy thật cùng phim: bắt đúng các ca trôi rõ (``张依贝→宋钱`` cùng cảnh: tự xưng
    em 11 / tôi 5, gọi anh 13 / cậu 11), không còn báo các ca đổi giọng giữa cảnh.

    Args:
        visual_cues: Chuỗi JSON Visual Cues (``id``/``spk``/``to``).
        source_by_index: Câu gốc theo chỉ số.
        translated_events: Các dòng đã dịch.

    Returns:
        Danh sách ``{"speaker", "addressee", "role", "from_index", "to_index", "variants",
        "examples"}``, nặng nhất trước.
    """
    translations = {int(e.index): (e.text or "").lower() for e in translated_events}
    by_pair = cue_lines_by_pair(visual_cues, set(translations))
    roles = (
        ("self", SELF_SOURCE_RE, YOU_SOURCE_RE),
        ("address", YOU_SOURCE_RE, SELF_SOURCE_RE),
    )
    findings: list[dict[str, Any]] = []
    for (speaker, addressee), indices in by_pair.items():
        for scene in split_scenes(indices):
            for role, wanted, excluded in roles:
                counts: dict[str, int] = {}
                examples: dict[str, int] = {}
                for index in scene:
                    source = source_by_index.get(index, "")
                    if (
                        PLURAL_SOURCE_RE.search(source)
                        or not wanted.search(source)
                        or excluded.search(source)
                    ):
                        continue
                    words = {word.lower() for word in ADDRESS_WORD_RE.findall(translations[index])}
                    if len(words) != 1:
                        continue  # nhiều đại từ trong một dòng → không biết đại từ nào ứng vai
                    word = next(iter(words))
                    counts[word] = counts.get(word, 0) + 1
                    examples.setdefault(word, index)
                total = sum(counts.values())
                if total < ADDRESS_DRIFT_MIN_LINES:
                    continue
                variants = {
                    word: count
                    for word, count in counts.items()
                    if count >= ADDRESS_DRIFT_MIN_EACH and count / total >= ADDRESS_DRIFT_MIN_SHARE
                }
                if len(variants) < 2:
                    continue
                findings.append({
                    "speaker": speaker,
                    "addressee": addressee,
                    "role": role,
                    "from_index": scene[0],
                    "to_index": scene[-1],
                    "variants": dict(sorted(variants.items(), key=lambda kv: -kv[1])),
                    "examples": {word: examples[word] for word in variants},
                })
    findings.sort(key=lambda item: -sum(item["variants"].values()))
    return findings
