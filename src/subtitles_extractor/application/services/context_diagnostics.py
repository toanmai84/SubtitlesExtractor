"""[v3.23.484] Chuẩn đoán NGỮ CẢNH TOÀN CỤC — đo chất lượng phân tích trước khi dịch.

Mục đích: xuất một báo cáo JSON GỌN, KHÔNG chứa toàn văn phụ đề, đủ để soi:

* Roster nhân vật có ĐỦ và có THỰC SỰ xuất hiện trong phụ đề không (bịa tên là dấu hiệu
  phân tích kém → dịch sai xưng hô).
* Glossary có bị trùng/rỗng/một chiều không.
* Visual cues có khớp chỉ số dòng thật không (cues trỏ sai dòng = vô dụng khi dịch).
* Độ phủ ngữ cảnh so với độ dài phim (tóm tắt quá ngắn cho phim dài = thiếu bối cảnh).

Toàn bộ hàm ở đây là HÀM THUẦN (không I/O, không phụ thuộc Qt/AI) để dễ kiểm thử và tái dùng.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from subtitles_extractor.application.services.cue_completion import (
    SpeakerCoverage,
    speaker_coverage,
)
from subtitles_extractor.application.services.glossary_reading_check import (
    ReadingOnlyTerm,
    find_reading_only_terms,
)
from subtitles_extractor.application.services.han_variants import fold_han_variants
from subtitles_extractor.application.services.name_reading_check import (
    SOURCE_ROSTER,
    NameReadingIssue,
    NameReadings,
    find_name_reading_issues,
)
from subtitles_extractor.application.services.roster_format import (
    MIN_CHARACTERS_FOR_ADDRESS_RULES,
    parse_roster,
    unread_roster_lines,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import TranslationLine

# Ngưỡng heuristic — đặt tên rõ để chỉnh về sau mà không phải dò trong code.
MIN_OVERVIEW_CHARS_PER_HOUR = 800
"""Tóm tắt tối thiểu (ký tự) cho mỗi giờ phim; thấp hơn → cảnh báo thiếu bối cảnh."""

MIN_CHARACTERS_FOR_DRAMA = 3
"""Phim truyện dưới ngần này nhân vật trong roster gần như chắc chắn là phân tích sót."""

MIN_GLOSSARY_TERMS_PER_1000_LINES = 4
"""Glossary tối thiểu cho mỗi 1000 dòng phụ đề; thấp hơn → nhắc bổ sung tên riêng/thuật ngữ."""

ROSTER_MENTION_SAMPLE_LIMIT = 60

#: [v3.23.559] Độ phủ Visual Cues. Từ v3.23.551 cues là NGUỒN DUY NHẤT cho biết ai nói
#: với ai, nên mọi lưới xưng hô (ép quy tắc, chuẩn hoá theo cảnh, chuẩn đoán trôi) chỉ
#: chạm được những dòng có cues. Đo thật cùng một phim: gemini-3.8-flash phủ 77-83%,
#: còn gemini-3.6-flash chỉ 27% — số cặp nhân vật được đo tụt 67 → 23 và các chỉ số
#: xưng hô "đẹp lên" một cách ẢO. Cờ cũ ``cues_sparse`` (chỉ báo dưới 2%, ghi "ít ảnh
#: hưởng") đã lỗi thời từ khi cues quyết định xưng hô.
CUES_COVERAGE_WARNING = 0.60
CUES_COVERAGE_ERROR = 0.40

#: Cửa sổ dòng để dò "VÙNG TRẮNG" cues — đoạn phim gần như không có manh mối.
CUES_DEAD_ZONE_WINDOW = 150
CUES_DEAD_ZONE_MAX_COVERAGE = 0.10
"""Số nhân vật tối đa đem đi đối chiếu với phụ đề (tránh chi phí O(n·m) khi roster khổng lồ)."""


class SubtitleEventLike(Protocol):
    """Chỉ cần ``index`` (1-based) và ``text`` — tránh phụ thuộc entity cụ thể."""

    index: int
    text: str


@dataclass(frozen=True, slots=True)
class ContextQualityFlag:
    """Một phát hiện về chất lượng ngữ cảnh.

    Attributes:
        code: Mã ngắn, ổn định để so sánh giữa các lần chạy (vd ``"roster_empty"``).
        severity: ``"info"`` | ``"warning"`` | ``"error"``.
        message: Mô tả tiếng Việt cho người dùng đọc.
    """

    code: str
    severity: str
    message: str


# Roster có cấu trúc do AI sinh thường ở dạng:
#   "林逸 (Lâm Dật) [nam, ~24 tuổi]: nhân vật chính, quyết đoán, mưu trí..."
# TÊN đứng trước phần chú thích ``(...)``/``[...]`` rồi tới dấu hai chấm. Regex dưới bắt ĐÚNG phần
# tên đó. Bắt buộc phải nhận diện theo dấu hai chấm, KHÔNG được tách roster theo dấu phẩy trước —
# vì mô tả cũng chứa dấu phẩy (v3.23.484 mắc lỗi này → cắt vụn mô tả thành "tên bịa").
_ROSTER_ENTRY_RE = re.compile(
    # Ranh giới trước TÊN: đầu chuỗi, xuống dòng, hoặc dấu kết câu/ngăn mục (. , ; • - …).
    # KHÔNG dùng dấu hai chấm làm ranh giới — vì mô tả thường có dấu hai chấm THỨ HAI
    # ("Tên [nam]: vai chính: mô tả") và ta không được coi "vai chính" là một tên.
    r"(?:^|[\n;•]|(?<=[.!?])\s|,\s)"
    # TÊN: không chứa dấu ngăn/kết câu/ngoặc → mảnh mô tả kiểu "hiếu thảo. 宋欣爽" không lọt.
    r"\s*[-–—*]?\s*([^,;\n:()\[\].!?]{1,40}?)\s*"
    r"(?:\([^)]*\)\s*)?(?:\[[^\]]*\]\s*)?:",
)

_MAX_NAME_WORDS = 5
"""Tên nhân vật hiếm khi quá 5 từ; dài hơn gần như chắc chắn là câu mô tả bị cắt nhầm."""


def _looks_like_name(candidate: str) -> bool:
    """Heuristic loại các mảnh MÔ TẢ bị cắt nhầm thành tên (vd 'quyết đoán', '~24 tuổi]').

    Dấu hiệu KHÔNG phải tên: kết thúc bằng dấu chấm (câu), quá nhiều từ, chứa ngoặc lẻ,
    hoặc là cụm tiếng Việt viết thường (tên riêng luôn viết hoa; tên CJK không có khái niệm
    hoa/thường nên được miễn kiểm tra này).
    """
    name = candidate.strip()
    if not name or len(name) > 40:
        return False
    if name.endswith((".", "!", "?")):
        return False
    if len(name.split()) > _MAX_NAME_WORDS:
        return False
    if any(ch in name for ch in "]})") and not any(ch in name for ch in "[({"):
        return False
    has_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" for ch in name)
    if has_cjk:
        return True
    first_alpha = next((ch for ch in name if ch.isalpha()), "")
    # Chữ Latin/Việt: tên riêng viết HOA chữ đầu; viết thường → là mô tả.
    return bool(first_alpha) and first_alpha.isupper()


def _split_roster(characters: str) -> list[str]:
    """Tách chuỗi roster thành danh sách TÊN (uỷ quyền cho ``roster_format`` — nguồn chân lý duy nhất).

    Từ v3.23.488, dữ liệu mới luôn ở QUY CÁCH CỐ ĐỊNH (mỗi nhân vật một dòng) nên đọc tất định;
    dữ liệu cũ / gõ tay vẫn có nhánh dự phòng heuristic bên trong ``roster_format``.
    """
    from subtitles_extractor.application.services.roster_format import roster_names

    return roster_names(characters)


def _split_glossary_terms(glossary: str) -> list[str]:
    """Tách glossary thành danh sách vế nguồn (phần trước dấu ``=``/``:``/``->``)."""
    terms: list[str] = []
    for line in glossary.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        source_side = re.split(r"\s*(?:=|:|->|→|\|)\s*", stripped, maxsplit=1)[0].strip()
        if source_side:
            terms.append(source_side)
    return terms


def _parse_visual_cues(visual_cues: str) -> dict[str, Any] | list[Any] | None:
    """Parse visual cues JSON; trả None nếu rỗng/không hợp lệ (KHÔNG raise)."""
    text = visual_cues.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict | list) else None


def _cue_line_indices(parsed_cues: dict[str, Any] | list[Any]) -> list[int]:
    """Rút các chỉ số dòng mà cues trỏ tới.

    Hỗ trợ mọi dạng ứng dụng đang dùng:

    * Bọc ngoài ``{"cues": [...]}`` — dạng ứng dụng lưu sau khi gộp các đoạn.
    * Danh sách bản ghi khoá RÚT GỌN ``{"id": 12, "spk": ..., "to": ..., "cue": ...}``
      (khoá ``id`` là dạng thật của ứng dụng — v3.23.484 bỏ sót nên báo nhầm "cues rỗng").
    * Danh sách bản ghi khoá dài ``index``/``line``/``line_index``/``idx``.
    * Từ điển theo chỉ số ``{"12": "..."}``.
    """
    if isinstance(parsed_cues, dict):
        for wrapper_key in ("cues", "visual_cues", "items", "data"):
            inner = parsed_cues.get(wrapper_key)
            if isinstance(inner, list | dict):
                return _cue_line_indices(inner)
        indices: list[int] = []
        for key in parsed_cues:
            try:
                indices.append(int(key))
            except (TypeError, ValueError):
                continue
        return indices
    indices = []
    for item in parsed_cues:
        if not isinstance(item, dict):
            continue
        for key in ("id", "index", "line", "line_index", "idx"):
            if key in item:
                try:
                    indices.append(int(item[key]))
                except (TypeError, ValueError):
                    pass
                break
    return indices


def _count_name_mentions(
    names: list[str], joined_text: str, cue_speakers: set[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Chia roster thành (có trong phụ đề, nhận diện QUA HÌNH ẢNH, không thấy ở đâu).

    Đối chiếu không phân biệt hoa/thường. Tên ghi kèm BÍ DANH (``"胖子 / 三胖子"``) được coi là
    xuất hiện khi BẤT KỲ bí danh nào có mặt.

    [v3.23.494] Nhiều nhân vật hợp lệ KHÔNG hề được gọi tên trong thoại — model nhận ra họ qua
    VIDEO và đặt nhãn mô tả (``绿发路人`` người qua đường tóc xanh, ``粉裙女同学`` bạn học váy hồng).
    Trước đây bị gộp chung vào "nghi AI bịa tên" (15 dương tính giả trên dữ liệu thật). Nay nếu
    nhãn đó CÓ xuất hiện trong Visual Cues thì xếp riêng: đó là bằng chứng model thật sự thấy
    nhân vật này trong hình, không phải bịa.

    Args:
        names: Danh sách tên trong roster.
        joined_text: Toàn văn phụ đề gốc để đối chiếu.
        cue_speakers: Tập nhãn người nói/người nghe xuất hiện trong Visual Cues.

    Returns:
        Bộ ba ``(mentioned, video_only, missing)``.

    [v3.23.615] So không phân biệt phồn/giản (``han_variants``): phụ đề OCR thật ghi
    "楚瑤" trong khi bảng nhân vật ghi "楚瑶" — nữ chính bị xếp "chỉ nhận ra qua hình".
    """
    lowered = fold_han_variants(joined_text.lower())
    speakers = {fold_han_variants(s.lower()) for s in (cue_speakers or set())}
    mentioned: list[str] = []
    video_only: list[str] = []
    missing: list[str] = []
    for name in names[:ROSTER_MENTION_SAMPLE_LIMIT]:
        aliases = [
            fold_han_variants(a.strip().lower())
            for a in re.split(r"\s*[/|]\s*", name)
            if a.strip()
        ]
        if any(alias in lowered for alias in aliases):
            mentioned.append(name)
        elif any(alias in speakers for alias in aliases):
            video_only.append(name)
        else:
            missing.append(name)
    return mentioned, video_only, missing


def _cue_speaker_labels(parsed_cues: dict[str, Any] | list[Any] | None) -> set[str]:
    """Rút tập nhãn người nói/người nghe từ Visual Cues (hỗ trợ khoá rút gọn ``spk``/``to``)."""
    if parsed_cues is None:
        return set()
    items: list[Any] = []
    if isinstance(parsed_cues, dict):
        for wrapper_key in ("cues", "visual_cues", "items", "data"):
            inner = parsed_cues.get(wrapper_key)
            if isinstance(inner, list):
                items = inner
                break
        else:
            items = [v for v in parsed_cues.values() if isinstance(v, dict)]
    else:
        items = list(parsed_cues)
    labels: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("spk", "speaker", "to", "addressee"):
            value = str(item.get(key) or "").strip()
            if value:
                labels.add(value)
    return labels


def _near_duplicate_pairs(
    entries: list[tuple[str, str]],
) -> list[tuple[str, str, bool]]:
    """[v3.23.494/496] Tìm cặp thuật ngữ gần trùng — cùng độ dài, khác ĐÚNG một ký tự, VÀ có
    bản dịch giống nhau.

    Vì sao phải xét bản dịch: chỉ so mặt chữ sẽ báo nhầm các cặp TRÁI NGHĨA. Dữ liệu thật có
    ``装货`` (chất hàng) và ``卸货`` (dỡ hàng) — khác một ký tự nhưng nghĩa ngược nhau, không phải
    trùng lặp. Ngược lại ``满月宴``/``满月酒`` cùng dịch là "tiệc đầy tháng" → đúng là một thực thể
    ghi hai cách.

    [v3.23.588] Phân biệt hai loại. Bản dịch GIỐNG HỆT (11/14 cặp của 17 bảng thật) thì
    hai mục cho cùng một chỉ thị — vô hại, và mỗi cách viết có mặt trong phụ đề đều cần
    mục của nó để được nhắc. Chỉ khi bản dịch KHÁC nhau (một bản chứa bản kia) thì khâu
    dịch mới nhận hai chỉ thị khác nhau cho một thực thể.

    [v3.23.592] "Chứa" phải đúng ở phần DÙNG TRONG LỜI THOẠI, không tính chú thích trong
    ngoặc. Bản phân tích thật có ``借種 = mượn giống (thụ tinh nhân tạo/nhờ người phối
    giống)`` và ``配種 = phối giống``: bị báo là một thực thể ghi hai cách chỉ vì chú
    thích của mục đầu nhắc tới "phối giống" — cảnh báo duy nhất về bảng thuật ngữ của bản
    phân tích đó.
    Trên 19 bảng thật: 4 cặp bị báo "bản dịch khác nhau", đây là cặp duy nhất khớp nhờ chú
    thích; ba cặp còn lại không đổi.

    Args:
        entries: Danh sách cặp ``(thuật ngữ gốc, bản dịch)``.

    Returns:
        Các cặp ``(thuật ngữ thứ nhất, thuật ngữ thứ hai, bản dịch có giống hệt không)``.
    """
    from subtitles_extractor.application.services.glossary_options import (
        split_glossary_note,
    )

    pairs: list[tuple[str, str, bool]] = []
    by_length: dict[int, list[tuple[str, str]]] = {}
    for term, translation in entries:
        by_length.setdefault(len(term), []).append((term, translation))
    for group in by_length.values():
        if len(group) < 2 or len(group) > 200:  # nhóm quá lớn → bỏ qua cho rẻ
            continue
        for i, (first, first_tr) in enumerate(group):
            for second, second_tr in group[i + 1 :]:
                diff = sum(1 for a, b in zip(first, second, strict=True) if a != b)
                if diff != 1:
                    continue
                left = first_tr.strip().lower()
                right = second_tr.strip().lower()
                # Thiếu bản dịch ở một bên → không đủ căn cứ, bỏ qua để tránh báo nhầm.
                if not left or not right:
                    continue
                if left == right:
                    pairs.append((first, second, True))
                    continue
                left_used = split_glossary_note(left)[0]
                right_used = split_glossary_note(right)[0]
                if (left in right or right in left) and (
                    left_used in right_used or right_used in left_used
                ):
                    pairs.append((first, second, False))
    return pairs


CJK_RANGE_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
"""Chữ Hán, Kana và Hangul — dùng để nhận ra tên còn ở ngôn ngữ GỐC."""


def _roster_names_in_target_language(names: list[str], source_lang: str) -> list[str]:
    """[v3.23.536] Tìm tên nhân vật bị ghi bằng NGÔN NGỮ ĐÍCH thay vì tên gốc.

    Quy tắc từ v3.23.490: trường ``name`` phải là tên GỐC nguyên văn, phần phiên âm để ở ``aka``.
    Dữ liệu thật vi phạm: roster có ``"Cậu bé lanh lợi (Cậu bé lanh lợi)"`` — cả hai vế đều tiếng
    Việt. Hậu quả dây chuyền: bảng quy đổi tên gửi cho khâu dịch không có gì để tra, nhãn người nói
    trong Visual Cues (vốn ghi bằng tên gốc) không khớp, và mốc neo căn dòng mất tác dụng với các
    nhân vật đó.

    Args:
        names: Danh sách tên trong roster.
        source_lang: Mã ngôn ngữ nguồn (chỉ xét khi nguồn dùng chữ CJK).

    Returns:
        Các tên không chứa ký tự nào của ngôn ngữ gốc.
    """
    if source_lang.lower()[:2] not in {"zh", "ja", "ko"}:
        return []
    flagged: list[str] = []
    for name in names:
        alias_text = re.split(r"\s*[/|]\s*", name)[0]
        if alias_text.strip() and not CJK_RANGE_RE.search(alias_text):
            flagged.append(name)
    return flagged


#: [v3.23.566] Dấu hiệu bộ gộp tóm tắt cũ đã cắt bỏ phần giữa phim (xem roster_format).
OVERVIEW_TRIM_MARKER = "lược bớt phần giữa"

#: Số từ đầu đoạn văn dùng để nhận ra các bản tóm tắt kể lại từ đầu.
_OPENING_WORDS = 5


def repeated_overview_openings(overview: str) -> int:
    """[v3.23.566] Số đoạn văn của tóm tắt có câu mở đầu trùng một đoạn khác (hàm thuần).

    Ca thật (gemini-3.8-flash, bộ gộp cũ): 7/13 đoạn văn mở đầu "Thế giới quan đan xen
    giữa cuộc sống nông thôn…" — mỗi đoạn video model kể lại toàn phim từ đầu.

    Args:
        overview: Tóm tắt phim.

    Returns:
        Tổng số đoạn văn thuộc các nhóm mở đầu trùng nhau (0 nếu không nhóm nào ≥ 3).
    """
    openings: dict[str, int] = {}
    for paragraph in re.split(r"\n\s*\n", overview or ""):
        words = paragraph.split()[:_OPENING_WORDS]
        if len(words) == _OPENING_WORDS:
            key = " ".join(words).casefold()
            openings[key] = openings.get(key, 0) + 1
    return sum(count for count in openings.values() if count >= 3)


def _usable_cue_indices(visual_cues: str, characters: str) -> list[int] | None:
    """[v3.23.562] Dòng có cue mang nhãn nhân vật trong bảng; None nếu không có bảng.

    Args:
        visual_cues: Chuỗi JSON Visual Cues.
        characters: Bảng nhân vật.

    Returns:
        Chỉ số dòng đã sắp xếp, hoặc None khi không có bảng nhân vật để đối chiếu.
    """
    from subtitles_extractor.application.services.cue_label_canonicalizer import (
        roster_label_map,
        usable_cue_lines,
    )
    from subtitles_extractor.application.services.visual_cue_serializer import (
        deserialize_visual_cues,
    )

    if not roster_label_map(characters):
        return None
    cues = deserialize_visual_cues((visual_cues or "").strip())
    return sorted(usable_cue_lines(cues, characters))


def cue_dead_zones(
    cue_indices: list[int],
    total_lines: int,
    window: int = CUES_DEAD_ZONE_WINDOW,
    max_coverage: float = CUES_DEAD_ZONE_MAX_COVERAGE,
) -> list[tuple[int, int]]:
    """[v3.23.559] Các khoảng dòng liền nhau gần như KHÔNG có Visual Cues (hàm thuần).

    Ca thật v3.23.558: cues phủ không đều — đoạn video 9 và 12 được 72-80%, trong khi đoạn
    19 chỉ 0,5% (1/208 dòng). Con số tổng 27% không cho biết chỗ nào hỏng; liệt kê vùng
    trắng giúp người dùng biết đoạn nào cần phân tích lại.

    Args:
        cue_indices: Chỉ số dòng (1-based) có cues.
        total_lines: Tổng số dòng phụ đề.
        window: Độ dài cửa sổ trượt (dòng).
        max_coverage: Độ phủ tối đa để coi là vùng trắng.

    Returns:
        Danh sách ``(dòng_đầu, dòng_cuối)`` đã gộp các cửa sổ chồng lấn.
    """
    if total_lines < window:
        return []
    covered = [0] * (total_lines + 2)
    for index in set(cue_indices):
        if 1 <= index <= total_lines:
            covered[index] = 1
    prefix = [0]
    for index in range(1, total_lines + 1):
        prefix.append(prefix[-1] + covered[index])
    zones: list[tuple[int, int]] = []
    for start in range(1, total_lines - window + 2):
        end = start + window - 1
        if (prefix[end] - prefix[start - 1]) / window > max_coverage:
            continue
        if zones and start <= zones[-1][1] + 1:
            zones[-1] = (zones[-1][0], end)
        else:
            zones.append((start, end))
    # Cửa sổ được phép chứa tới ``max_coverage`` dòng có cues nên mép vùng lấn vào phần
    # đã phủ — cắt mép về dòng trống đầu/cuối để người dùng thấy đúng chỗ hỏng.
    trimmed: list[tuple[int, int]] = []
    for zone_start, zone_end in zones:
        while zone_start < zone_end and covered[zone_start]:
            zone_start += 1
        while zone_end > zone_start and covered[zone_end]:
            zone_end -= 1
        trimmed.append((zone_start, zone_end))
    return trimmed


#: [v3.23.568] Dưới tỉ lệ này số mệnh đề xưng hô tra được người nghe thì báo — ca thật
#: gemini-3.6-flash ghi người nghe bằng tên gốc, bộ tra cũ chỉ đọc được 6/58 mệnh đề.
MIN_RESOLVED_ADDRESS_SHARE = 0.8
#: Chỉ xét khi bảng có đủ số mệnh đề để tỉ lệ có nghĩa.
MIN_ADDRESS_CLAUSES_FOR_FLAG = 5


def _vocative_cue_label_fixes(
    visual_cues: str, events: list[Any], characters: str
) -> tuple[list[int], list[int], list[int]]:
    """[v3.23.570] Dòng gọi tên có nhãn cues sai mà khâu dịch sẽ tự sửa (hàm thuần).

    Returns:
        (dòng bị ghi ngược người nói/nghe, [v3.23.571] dòng gọi kèm vai vế có người nói
        trái vai, [v3.23.603] dòng TRẢ LỜI tên ghi sai người nói) — cùng thứ tự sửa với
        khâu dịch.
    """
    if not visual_cues.strip() or not characters.strip() or not events:
        return [], [], []
    from subtitles_extractor.application.services.visual_cue_serializer import (
        deserialize_visual_cues,
    )
    from subtitles_extractor.application.services.vocative_guard import (
        fix_vocative_cue_labels,
    )

    # Cues đánh số theo VỊ TRÍ dòng (1..N), giống khâu dịch.
    sources = {
        position: str(getattr(event, "text", "") or "")
        for position, event in enumerate(events, start=1)
    }
    fixes = fix_vocative_cue_labels(
        deserialize_visual_cues(visual_cues), sources, characters
    )
    return fixes.swapped, fixes.kin_relabeled, fixes.name_answers


def _cue_run_fill_lines(visual_cues: str, events: list[Any]) -> int:
    """[v3.23.572] Số dòng nói tiếp mà khâu dịch sẽ tự điền nhãn cues (hàm thuần)."""
    if not visual_cues.strip() or not events:
        return 0
    from subtitles_extractor.application.services.cue_run_filler import (
        LineSpan,
        fill_cue_runs,
    )
    from subtitles_extractor.application.services.visual_cue_serializer import (
        deserialize_visual_cues,
    )

    # Thiếu mốc thời gian (sự kiện giả trong test) thì coi như không có khoảng lặng.
    spans = [
        LineSpan(
            position,
            round(float(getattr(event, "start_sec", 0.0) or 0.0) * 1000),
            round(float(getattr(event, "end_sec", 0.0) or 0.0) * 1000),
        )
        for position, event in enumerate(events, start=1)
    ]
    _, filled = fill_cue_runs(deserialize_visual_cues(visual_cues), spans)
    return len(filled)


def _address_rule_stats(characters: str) -> dict[str, Any]:
    """[v3.23.568] Số mệnh đề xưng hô và số tra được người nghe (hàm thuần)."""
    from subtitles_extractor.application.services.roster_format import address_clauses

    clauses = address_clauses(characters)
    unresolved = [c.raw_listener for c in clauses if not c.listeners]
    return {
        "clauses": len(clauses),
        "resolved": len(clauses) - len(unresolved),
        "unresolved_listeners": list(dict.fromkeys(unresolved))[:20],
    }


def _address_rule_flags(stats: dict[str, Any]) -> list[ContextQualityFlag]:
    """[v3.23.568] Cờ khi phần lớn quy tắc xưng hô không gắn được vào cặp nhân vật nào."""
    total = int(stats["clauses"])
    if total < MIN_ADDRESS_CLAUSES_FOR_FLAG:
        return []
    share = int(stats["resolved"]) / total
    if share >= MIN_RESOLVED_ADDRESS_SHARE:
        return []
    sample = ", ".join(stats["unresolved_listeners"][:8])
    return [ContextQualityFlag(
        "address_rules_unresolved", "warning",
        f"Chỉ {stats['resolved']}/{total} quy tắc xưng hô tra được người nghe trong bảng "
        f"nhân vật ({share:.0%}) — các quy tắc còn lại KHÔNG được gắn vào dòng dịch và "
        f"lưới xưng hô không kiểm được. Người nghe không tra được: {sample}.",
    )]


def _roster_reading_flags(
    unread_lines: list[str], roster_size: int, address_stats: dict[str, Any]
) -> list[ContextQualityFlag]:
    """[v3.23.614] Cờ khi bảng nhân vật đọc thiếu dòng, hoặc đọc được mà không có xưng hô.

    Ca thật 14/10: bảng 7 người chỉ đọc được 4 tên trơn, 0 quy tắc xưng hô, mà chẩn đoán
    ghi ``quality_flags: []``. Hai cờ này làm lỗi kiểu đó lộ ra ngay ở file chẩn đoán.
    """
    flags: list[ContextQualityFlag] = []
    if unread_lines:
        sample = "; ".join(line.split(" ", 1)[0] for line in unread_lines[:6])
        flags.append(ContextQualityFlag(
            "roster_lines_unread", "error",
            f"Bộ đọc bảng nhân vật bỏ sót {len(unread_lines)} dòng ({sample}) — các "
            "nhân vật này mất giới tính, phiên âm và quy tắc xưng hô ở mọi khâu sau "
            "(dịch, lồng tiếng). Đây là lỗi của ứng dụng, hãy gửi file chẩn đoán này.",
        ))
    if roster_size >= MIN_CHARACTERS_FOR_ADDRESS_RULES and not address_stats["clauses"]:
        flags.append(ContextQualityFlag(
            "roster_without_address_rules", "warning",
            f"Bảng {roster_size} nhân vật không có quy tắc xưng hô nào — khâu dịch tự "
            "chọn xưng hô từng lô (dễ đổi giữa chừng); nên bấm Phân tích lại.",
        ))
    return flags


#: Số mục lệch âm tối đa nêu trong thông điệp cờ (đủ để soát, không làm dài file).
NAME_READING_MESSAGE_LIMIT = 6


READING_ONLY_MESSAGE_LIMIT = 4
"""Số mục "chỉ đọc âm" nêu làm ví dụ trong thông báo."""


def _reading_only_flags(terms: list[ReadingOnlyTerm]) -> list[ContextQualityFlag]:
    """[v3.23.592] Cờ mục thuật ngữ chỉ ĐỌC ÂM, cách nói dễ hiểu nằm trong ngoặc.

    Mức ``warning`` vì các mục này đi thẳng vào lời thoại: đo trên 56 lượt dịch thật,
    91% số dòng chứa thuật ngữ như vậy mang nguyên phần đọc âm ở bản cuối.
    """
    if not terms:
        return []
    shown = "; ".join(term.describe() for term in terms[:READING_ONLY_MESSAGE_LIMIT])
    more = len(terms) - READING_ONLY_MESSAGE_LIMIT
    return [ContextQualityFlag(
        "glossary_reading_only", "warning",
        f"{len(terms)} mục thuật ngữ chỉ ĐỌC ÂM Hán-Việt từng chữ, còn cách nói dễ hiểu "
        f"lại nằm trong ngoặc: {shown}{f'; … và {more} mục nữa' if more > 0 else ''}. "
        "Khâu dịch chép phần TRƯỚC ngoặc vào lời thoại. Mục nào phần trong ngoặc mới là "
        "cách người Việt nói thì sửa tay trong bảng thuật ngữ (đưa cách nói đó ra trước "
        "ngoặc); từ Hán-Việt vẫn dùng thường ngày thì để nguyên.",
    )]


def _name_reading_flags(issues: list[NameReadingIssue]) -> list[ContextQualityFlag]:
    """[v3.23.567] Cờ phiên âm tên KHÁC âm Hán-Việt (hàm thuần).

    Bảng nhân vật → ``warning`` (tên sai lan ra mọi dòng thoại); bảng thuật ngữ → ``info``
    (nhiều mục dịch theo nghĩa/âm quy ước ngành một cách hợp lệ, chỉ để soát tay).
    """

    def _describe(group: list[NameReadingIssue]) -> str:
        shown = [
            f'{i.name} "{i.written}" ({i.char} đọc {"/".join(i.expected)})'
            + (f' → tự sửa "{i.suggestion}"' if i.suggestion else "")
            for i in group[:NAME_READING_MESSAGE_LIMIT]
        ]
        more = len(group) - len(shown)
        return "; ".join(shown) + (f"; … và {more} mục nữa" if more > 0 else "")

    roster_issues = [i for i in issues if i.source == SOURCE_ROSTER]
    glossary_issues = [i for i in issues if i.source != SOURCE_ROSTER]
    flags: list[ContextQualityFlag] = []
    if roster_issues:
        flags.append(ContextQualityFlag(
            "roster_name_reading", "warning",
            f"{len(roster_issues)} chữ trong tên nhân vật được phiên âm KHÁC âm "
            f"Hán-Việt: {_describe(roster_issues)}. Mục có 'tự sửa' được ứng dụng sửa "
            "tự động khi mở lại video/khi dịch; mục còn lại cần soát tay (có thể là tên "
            "Việt hoá có chủ ý).",
        ))
    if glossary_issues:
        flags.append(ContextQualityFlag(
            "glossary_name_reading", "info",
            f"{len(glossary_issues)} chữ trong bảng thuật ngữ được phiên âm khác âm "
            f"Hán-Việt: {_describe(glossary_issues)}. Chỉ để soát tay — thuật ngữ hay "
            "dịch theo nghĩa hoặc theo âm quy ước.",
        ))
    return flags


def _speaker_coverage(visual_cues: str, events: list[Any]) -> SpeakerCoverage:
    """[v3.23.600] Độ phủ người nói của các dòng có lời (xem ``cue_completion``)."""
    lines = [
        TranslationLine(
            index=int(getattr(event, "index", 0)), start_ms=0, end_ms=0,
            text=str(getattr(event, "text", "") or ""),
        )
        for event in events
    ]
    return speaker_coverage(deserialize_visual_cues(visual_cues), lines)


def _missing_speaker_flag(missing: tuple[int, ...], spoken: int) -> ContextQualityFlag:
    """[v3.23.600] Cờ: còn dòng có lời chưa có người nói."""
    sample = ", ".join(str(line) for line in missing[:8])
    return ContextQualityFlag(
        "speakers_missing", "warning",
        f"{len(missing)}/{spoken} dòng có lời CHƯA có người nói (vd dòng {sample}) — "
        "lồng tiếng sẽ đọc các dòng này bằng giọng chung. Bấm '🎭 Bổ sung người nói' "
        "(trang Dịch, thẻ Ngữ cảnh) để hỏi lại riêng các dòng đó — phân tích khôi phục từ "
        "phiên cũ không tự hỏi lại; khâu dịch có kèm video cũng tự hỏi. Còn thiếu thì soát "
        "ở trang TTS (⚠ Soát người nói).",
    )


def _quality_flags(
    *,
    overview: str,
    roster: list[str],
    missing_names: list[str],
    glossary_terms: list[str],
    duplicate_terms: list[str],
    near_duplicate_terms: list[str],
    conflicting_near_duplicates: list[str],
    half_translated_terms: list[str],
    names_in_target_language: list[str],
    ambiguous_address: list[str] | None = None,
    cue_indices: list[int],
    out_of_range_cues: list[int],
    total_lines: int,
    duration_sec: float,
    attach_video: bool = False,
    usable_cue_indices: list[int] | None = None,
) -> list[ContextQualityFlag]:
    """Sinh danh sách cờ chất lượng từ các số liệu đã đo (hàm thuần)."""
    flags: list[ContextQualityFlag] = []
    if not overview.strip():
        flags.append(ContextQualityFlag(
            "overview_empty", "error",
            "Tóm tắt bối cảnh RỖNG — AI dịch sẽ thiếu mạch truyện.",
        ))
    elif duration_sec > 0:
        hours = duration_sec / 3600.0
        min_chars = int(MIN_OVERVIEW_CHARS_PER_HOUR * max(1.0, hours))
        if len(overview) < min_chars:
            flags.append(ContextQualityFlag(
                "overview_thin", "warning",
                f"Tóm tắt chỉ {len(overview)} ký tự cho ~{hours:.1f} giờ phim "
                f"(khuyến nghị ≥ {min_chars}) — có thể thiếu bối cảnh nửa sau phim.",
            ))
    repeated = repeated_overview_openings(overview)
    if repeated:
        flags.append(ContextQualityFlag(
            "overview_repetitive", "warning",
            f"Tóm tắt có {repeated} đoạn văn mở đầu gần như y hệt nhau — model kể lại toàn "
            "phim từ đầu ở mỗi đoạn video và các bản bị NỐI vào nhau. Phần lặp chiếm chỗ "
            "của diễn biến thật; chạy lại phân tích (bản mới ghi diễn biến theo từng đoạn).",
        ))
    if OVERVIEW_TRIM_MARKER in overview:
        flags.append(ContextQualityFlag(
            "overview_truncated", "warning",
            "Tóm tắt đã chạm trần độ dài và bị CẮT BỎ PHẦN GIỮA PHIM — các lô dịch ở giữa "
            "phim thiếu bối cảnh. Chạy lại phân tích để có tóm tắt theo từng đoạn thời gian.",
        ))
    if not roster:
        flags.append(ContextQualityFlag(
            "roster_empty", "error",
            "Bảng nhân vật RỖNG — dịch dễ sai xưng hô (anh/chị/em/ông/bà).",
        ))
    elif len(roster) < MIN_CHARACTERS_FOR_DRAMA:
        flags.append(ContextQualityFlag(
            "roster_thin", "warning",
            f"Bảng nhân vật chỉ có {len(roster)} tên — nhiều khả năng phân tích còn sót.",
        ))
    if missing_names:
        flags.append(ContextQualityFlag(
            "roster_names_not_in_subtitles", "warning",
            f"{len(missing_names)} tên trong roster KHÔNG xuất hiện ở phụ đề gốc "
            f"(vd: {', '.join(missing_names[:5])}) — nghi AI bịa tên hoặc lấy nhầm nguồn.",
        ))
    if duplicate_terms:
        flags.append(ContextQualityFlag(
            "glossary_duplicates", "warning",
            f"{len(duplicate_terms)} thuật ngữ TRÙNG trong glossary "
            f"(vd: {', '.join(duplicate_terms[:5])}) — dễ gây dịch không nhất quán.",
        ))
    if names_in_target_language:
        flags.append(ContextQualityFlag(
            "roster_names_not_in_source_language", "warning",
            f"{len(names_in_target_language)} tên trong bảng nhân vật được ghi bằng NGÔN NGỮ ĐÍCH "
            f"thay vì tên gốc (vd: {', '.join(names_in_target_language[:3])}) — bảng quy đổi tên "
            "gửi cho khâu dịch sẽ không tra được, và nhãn người nói trong Visual Cues cũng không "
            "khớp. Nên sửa lại thành tên gốc, phần tiếng Việt để trong ngoặc.",
        ))
    if ambiguous_address:
        flags.append(ContextQualityFlag(
            "roster_ambiguous_address", "warning",
            f"{len(ambiguous_address)} cặp nhân vật có bảng xưng hô MƠ HỒ — liệt kê nhiều đại từ "
            f"cho cùng một cặp (vd: {'; '.join(ambiguous_address[:2])}). Khâu dịch được phép chọn "
            "bất kỳ nên mỗi lô chọn một kiểu, xưng hô trôi ngay trong cùng cảnh. Nên phân tích "
            "lại ngữ cảnh (prompt đã yêu cầu chốt MỘT cặp đại từ) hoặc sửa tay bảng nhân vật.",
        ))
    if half_translated_terms:
        flags.append(ContextQualityFlag(
            "glossary_half_translated", "error",
            f"{len(half_translated_terms)} mục thuật ngữ DỊCH DỞ — vế dịch còn lẫn chữ Hán "
            f"(vd: {', '.join(half_translated_terms[:3])}). Mục hỏng này đi thẳng vào prompt "
            "dịch và DẠY model để sót chữ Hán trong bản tiếng Việt. Khâu dịch đã tự loại chúng, "
            "nhưng nên phân tích lại ngữ cảnh để có bảng thuật ngữ đầy đủ.",
        ))
    if conflicting_near_duplicates:
        flags.append(ContextQualityFlag(
            "glossary_near_duplicates", "warning",
            f"{len(conflicting_near_duplicates)} cặp thuật ngữ GẦN TRÙNG (khác 1 ký tự, "
            f"vd: {', '.join(conflicting_near_duplicates[:3])}) có bản dịch KHÁC nhau — "
            "nhiều khả năng là CÙNG một thực thể ghi hai cách; khâu dịch sẽ nhận hai chỉ "
            "thị khác nhau.",
        ))
    elif near_duplicate_terms:
        # [v3.23.588] Bản dịch giống hệt: hai mục cho CÙNG một chỉ thị, không có gì mâu
        # thuẫn. Trước đây vẫn báo "cảnh báo… hai chỉ thị mâu thuẫn" — cảnh báo duy nhất
        # của một bản phân tích sạch, mà người dùng không có gì để sửa.
        flags.append(ContextQualityFlag(
            "glossary_near_duplicates", "info",
            f"{len(near_duplicate_terms)} cặp thuật ngữ GẦN TRÙNG (khác 1 ký tự, vd: "
            f"{', '.join(near_duplicate_terms[:3])}) có CÙNG bản dịch — một thực thể ghi "
            "hai cách; không ảnh hưởng bản dịch.",
        ))
    if not glossary_terms:
        flags.append(ContextQualityFlag(
            "glossary_empty", "info",
            "Glossary rỗng — chấp nhận được với phim đời thường, nhưng phim "
            "tiên hiệp/võ hiệp nên có để thống nhất thuật ngữ.",
        ))
    elif total_lines >= 1000 and len(glossary_terms) < MIN_GLOSSARY_TERMS_PER_1000_LINES * (
        total_lines / 1000.0
    ):
        flags.append(ContextQualityFlag(
            "glossary_thin", "info",
            f"Glossary chỉ {len(glossary_terms)} thuật ngữ cho {total_lines} dòng — mỏng. "
            "Tên riêng/địa danh/chức danh lặp lại nên đưa vào để dịch nhất quán giữa các tập.",
        ))
    if attach_video and not cue_indices:
        flags.append(ContextQualityFlag(
            "cues_empty_despite_video", "warning",
            "Có ĐÍNH VIDEO khi phân tích nhưng Visual Cues RỖNG — phần phân tích hình ảnh không "
            "tạo được manh mối nào (kiểm tra: đoạn video có tải lên đủ không, model có hỗ trợ "
            "hình ảnh không, hoặc bật lại tuỳ chọn phân tích Visual Cues).",
        ))
    if out_of_range_cues:
        flags.append(ContextQualityFlag(
            "cues_out_of_range", "error",
            f"{len(out_of_range_cues)} visual cue trỏ tới dòng KHÔNG tồn tại "
            f"(vd: {out_of_range_cues[:5]}) — cues lệch chỉ số, vô dụng khi dịch.",
        ))
    if total_lines > 0 and cue_indices:
        # [v3.23.562] Đo trên dòng có cue DÙNG ĐƯỢC (nhãn là nhân vật trong bảng). Ca
        # thật: cues "phủ" 99,7% nhưng 74% chỉ mang nhãn chung chung ("Nữ tỳ"…).
        measured = cue_indices if usable_cue_indices is None else usable_cue_indices
        coverage = len(set(measured)) / total_lines
        labeled = len(set(cue_indices)) / total_lines
        if coverage < CUES_COVERAGE_WARNING:
            zones = cue_dead_zones(measured, total_lines)
            generic_note = (
                f" Có cues ở {labeled * 100:.0f}% số dòng nhưng phần lớn chỉ mang nhãn "
                "vai vế chung chung, KHÔNG khớp bảng nhân vật — lưới xưng hô không dùng "
                "được."
                if labeled - coverage >= 0.2
                else ""
            )
            zone_text = (
                f" Vùng gần như trắng: {', '.join(f'{a}-{b}' for a, b in zones[:6])}."
                if zones else ""
            )
            flags.append(ContextQualityFlag(
                "cues_low_coverage",
                "error" if coverage < CUES_COVERAGE_ERROR else "warning",
                f"Visual Cues chỉ phủ {coverage * 100:.0f}% số dòng (lần phân tích tốt "
                f"đạt ~80%).{generic_note}{zone_text} Cues là nguồn DUY NHẤT cho biết ai "
                "nói với ai, nên xưng hô ở các dòng thiếu cues sẽ không được lưới nào "
                "kiểm — và số liệu xưng hô trong chuẩn đoán dịch trông TỐT HƠN THỰC TẾ "
                "vì bị đo trên ít dòng hơn. "
                "Khi dịch có bật 'Phân tích hình ảnh: ai nói / nói với ai' và có video, "
                "ứng dụng tự quét bổ sung các dòng còn thiếu; muốn phân tích tốt hơn thì "
                "chạy lại phân tích bằng model mạnh hơn.",
            ))
    return flags


def build_context_diagnostics(
    *,
    context: Any,
    source_events: list[SubtitleEventLike] | None = None,
    video_path: str = "",
    duration_sec: float = 0.0,
    app_version: str = "",
    analysis_settings: dict[str, Any] | None = None,
    name_readings: NameReadings | None = None,
) -> dict[str, Any]:
    """Dựng báo cáo chuẩn đoán ngữ cảnh toàn cục (hàm thuần, KHÔNG I/O).

    Args:
        context: Đối tượng có các thuộc tính ``target_lang``/``source_lang``/``overview``/
            ``characters``/``glossary``/``visual_cues`` (vd ``TranslationContext``).
        source_events: Phụ đề gốc để đối chiếu roster/cues. Có thể None nếu chưa nạp.
        video_path: Đường dẫn video (chỉ lấy TÊN file để tránh lộ đường dẫn cá nhân).
        duration_sec: Thời lượng phim, dùng đánh giá độ dày tóm tắt.
        app_version: Phiên bản ứng dụng để đối chiếu giữa các lần xuất.
        analysis_settings: Cấu hình phân tích (model, fps, resolution…) để tương quan
            chất lượng ↔ cài đặt.
        name_readings: [v3.23.567] Bảng âm Hán-Việt để soát phiên âm tên (None = bỏ qua).

    Returns:
        Dict JSON-serializable gồm ``summary``, ``roster``, ``glossary``, ``visual_cues``
        và ``quality_flags``. KHÔNG chứa toàn văn phụ đề hay tóm tắt (chỉ trích đoạn ngắn).
    """
    events = list(source_events or [])
    overview = str(getattr(context, "overview", "") or "")
    characters = str(getattr(context, "characters", "") or "")
    glossary = str(getattr(context, "glossary", "") or "")
    visual_cues = str(getattr(context, "visual_cues", "") or "")

    roster = _split_roster(characters)
    joined_text = "\n".join(getattr(ev, "text", "") or "" for ev in events)
    parsed_cues_for_names = _parse_visual_cues(visual_cues)
    mentioned, video_only, missing = _count_name_mentions(
        roster, joined_text, _cue_speaker_labels(parsed_cues_for_names)
    )

    glossary_terms = _split_glossary_terms(glossary)
    seen_terms: set[str] = set()
    duplicate_terms: list[str] = []
    for term in glossary_terms:
        key = term.lower()
        if key in seen_terms:
            duplicate_terms.append(term)
        else:
            seen_terms.add(key)

    parsed_cues = _parse_visual_cues(visual_cues)
    cue_indices = _cue_line_indices(parsed_cues) if parsed_cues is not None else []
    usable_indices = _usable_cue_indices(
        visual_cues, str(getattr(context, "characters", "") or "")
    )
    valid_indices = {int(getattr(ev, "index", 0)) for ev in events}
    reversed_vocatives, kin_conflicts, name_answers = _vocative_cue_label_fixes(
        visual_cues, events, characters
    )
    run_fill_lines = _cue_run_fill_lines(visual_cues, events)
    speakers = _speaker_coverage(visual_cues, events)
    out_of_range = (
        sorted({i for i in cue_indices if i not in valid_indices}) if valid_indices else []
    )

    # [v3.23.496] Cần CẢ bản dịch để loại dương tính giả (cặp trái nghĩa khác 1 ký tự).
    from subtitles_extractor.application.services.roster_format import parse_glossary

    glossary_pairs = [(e.term, e.translation) for e in parse_glossary(glossary)]
    near_duplicate_pairs = _near_duplicate_pairs(glossary_pairs)
    near_duplicate_terms = [f"{a} ~ {b}" for a, b, _same in near_duplicate_pairs]
    conflicting_near_duplicates = [
        f"{a} ~ {b}" for a, b, same in near_duplicate_pairs if not same
    ]
    # [v3.23.549] Mục DỊCH DỞ: vế dịch lẫn chữ Hán với chữ Việt (vd ``生死道消 = "thân tử đạo消"``).
    # Ca thật lọt qua MỌI bộ kiểm rồi đi thẳng vào prompt dịch, dạy model để sót chữ Hán.
    from subtitles_extractor.application.services.roster_format import is_half_translated

    half_translated_terms = [
        f"{term} = {translation}"
        for term, translation in glossary_pairs
        if is_half_translated(translation)
    ]
    # [v3.23.550] Bảng xưng hô liệt kê NHIỀU đại từ cho một cặp → khâu dịch tự chọn mỗi lô một kiểu.
    from subtitles_extractor.application.services.roster_format import ambiguous_address_rules

    ambiguous_address = ambiguous_address_rules(
        str(getattr(context, "characters", "") or "")
    )
    # [v3.23.593] Nhân vật CHỈ nhận ra qua hình ảnh không có tên nào trong phụ đề để ghi
    # nguyên văn — nhãn vai trò tiếng đích ("Cậu bé trong thôn") là đúng, và nhãn người
    # nói trong Visual Cues dùng chính nhãn đó. Trên kho: 8/9 tên từng bị cờ này báo là
    # nhãn như vậy (tên thứ chín là mảnh vỡ "Xưng hô"), không tên thật nào.
    video_only_names = set(video_only)
    names_in_target_language = _roster_names_in_target_language(
        [name for name in roster if name not in video_only_names],
        str(getattr(context, "source_lang", "") or ""),
    )
    flags = _quality_flags(
        half_translated_terms=half_translated_terms,
        ambiguous_address=ambiguous_address,
        overview=overview,
        roster=roster,
        missing_names=missing,
        glossary_terms=glossary_terms,
        duplicate_terms=duplicate_terms,
        near_duplicate_terms=near_duplicate_terms,
        conflicting_near_duplicates=conflicting_near_duplicates,
        names_in_target_language=names_in_target_language,
        cue_indices=cue_indices,
        usable_cue_indices=usable_indices,
        out_of_range_cues=out_of_range,
        total_lines=len(events),
        duration_sec=duration_sec,
        attach_video=bool((analysis_settings or {}).get("attach_video", False)),
    )
    name_issues = (
        find_name_reading_issues(characters, glossary, name_readings)
        if name_readings
        else []
    )
    flags.extend(_name_reading_flags(name_issues))
    reading_only = (
        find_reading_only_terms(glossary, name_readings) if name_readings else []
    )
    flags.extend(_reading_only_flags(reading_only))
    address_stats = _address_rule_stats(characters)
    flags.extend(_address_rule_flags(address_stats))
    roster_entries = parse_roster(characters)
    unread_lines = unread_roster_lines(characters, roster_entries)
    flags.extend(_roster_reading_flags(unread_lines, len(roster_entries), address_stats))
    if visual_cues.strip() and speakers.missing:
        flags.append(_missing_speaker_flag(speakers.missing, speakers.spoken_lines))

    return {
        "schema": "subtitles_extractor.context_diagnostics/1",
        "app_version": app_version,
        "video_name": video_path.replace("\\", "/").rsplit("/", 1)[-1] if video_path else "",
        "duration_sec": round(float(duration_sec), 2),
        "analysis_settings": dict(analysis_settings or {}),
        "summary": {
            "target_lang": str(getattr(context, "target_lang", "") or ""),
            "source_lang": str(getattr(context, "source_lang", "") or ""),
            "subtitle_lines": len(events),
            "overview_chars": len(overview),
            "overview_excerpt": overview[:280],
            "roster_count": len(roster),
            "glossary_terms": len(glossary_terms),
            "visual_cues_lines": len(set(cue_indices)),
            "enable_tags": bool(getattr(context, "enable_tags", False)),
            "include_desc": bool(getattr(context, "include_desc", True)),
        },
        "roster": {
            "names": roster[:ROSTER_MENTION_SAMPLE_LIMIT],
            "mentioned_in_subtitles": mentioned,
            # [v3.23.494] Nhân vật model nhận ra QUA HÌNH ẢNH (không được gọi tên trong thoại)
            # — hợp lệ, không phải bịa tên.
            "identified_from_video": video_only,
            "not_found_in_subtitles": missing,
            "ambiguous_address": ambiguous_address[:30],
            "name_reading_issues": [
                issue.as_dict() for issue in name_issues if issue.source == SOURCE_ROSTER
            ],
            "address_rules": address_stats,
            # [v3.23.614] Dòng của bảng mà bộ đọc bỏ sót (phải luôn rỗng).
            "unread_lines": [line[:120] for line in unread_lines[:20]],
        },
        "glossary": {
            "terms_sample": glossary_terms[:50],
            "duplicates": duplicate_terms[:20],
            "near_duplicates": near_duplicate_terms[:20],
            "half_translated": half_translated_terms[:20],
            # [v3.23.592] Mục chỉ đọc âm Hán-Việt, cách nói dễ hiểu nằm trong ngoặc.
            "reading_only": [term.describe() for term in reading_only[:20]],
            "name_reading_issues": [
                issue.as_dict() for issue in name_issues if issue.source != SOURCE_ROSTER
            ][:20],
        },
        "visual_cues": {
            "parsed": parsed_cues is not None,
            "covered_lines": len(set(cue_indices)),
            "coverage_ratio": (
                round(len(set(cue_indices)) / len(events), 4) if events else 0.0
            ),
            # [v3.23.562] Dòng có cue mà người nói/nghe là nhân vật TRONG BẢNG — phần
            # lưới xưng hô thực sự dùng được (None khi không có bảng nhân vật).
            "roster_labeled_lines": (
                len(usable_indices) if usable_indices is not None else None
            ),
            "roster_labeled_ratio": (
                round(len(usable_indices) / len(events), 4)
                if usable_indices is not None and events
                else None
            ),
            "out_of_range_indices": out_of_range[:20],
            # [v3.23.570] Dòng GỌI TÊN mà nhãn ghi người được gọi là NGƯỜI NÓI — khâu dịch
            # tự đảo lại, số này đo độ tin của nhãn từ model phân tích.
            "reversed_vocative_lines": reversed_vocatives[:20],
            # [v3.23.571] Tiếng gọi kèm vai vế (``宋钱弟弟``) mà nhãn người nói TRÁI VAI
            # với hậu tố — khâu dịch đổi sang người nói khớp vai ở dòng kề.
            "kin_speaker_conflict_lines": kin_conflicts[:20],
            # [v3.23.603] Dòng TRẢ LỜI tên ("你叫甚麼名字" → "秦可秀") mà nhãn ghi là lời
            # người khác — khâu dịch và lồng tiếng tự sửa về người mang tên.
            "name_answer_lines": name_answers[:20],
            # [v3.23.572] Dòng NÓI TIẾP không có cue vì model chỉ ghi đầu lượt lời — khâu
            # dịch tự điền nhãn của dòng đầu lượt lời cho chúng.
            "run_fill_lines": run_fill_lines,
            # [v3.23.600] NGƯỜI NÓI của dòng có lời — lồng tiếng cần đủ 100%.
            "speaker_lines": speakers.with_speaker,
            "speaker_ratio": round(speakers.ratio, 4),
            "voice_only_speaker_lines": speakers.voice_only,
            "missing_speaker_lines": list(speakers.missing[:30]),
        },
        "quality_flags": [
            {"code": f.code, "severity": f.severity, "message": f.message} for f in flags
        ],
        "counts": {
            "errors": sum(1 for f in flags if f.severity == "error"),
            "warnings": sum(1 for f in flags if f.severity == "warning"),
        },
    }
