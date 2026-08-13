"""Translation Memory (TM) — truy hồi câu đã dịch để làm tham chiếu (RAG grounding).

[v3.23.55] Theo best practice dịch LLM cấp doanh nghiệp (translated.com, lokalise), việc
"grounding" đầu ra bằng các câu đã được dịch và phê duyệt trước đó (Translation Memory)
là cách mạnh nhất để TĂNG NHẤT QUÁN và GIẢM ảo giác thuật ngữ — đặc biệt quan trọng khi
dịch PHIM BỘ NHIỀU TẬP, nơi tên nhân vật, thuật ngữ và lối nói lặp lại xuyên suốt.

Module này chứa logic THUẦN (pure) cho:
- Biểu diễn một mục TM (``TranslationMemoryEntry``).
- Truy hồi các mục liên quan nhất tới một câu nguồn (``retrieve_relevant``) bằng độ tương
  đồng chuỗi (rapidfuzz), không phụ thuộc I/O hay cơ sở dữ liệu.
- Định dạng các mục đã truy hồi thành khối tham chiếu để chèn vào prompt
  (``format_reference_block``).

Việc lưu trữ bền vững nằm ở tầng infrastructure (SqliteTranslationMemoryStore).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

# Dấu phân tách giữa thuật ngữ gốc và bản dịch trong một dòng glossary.
_SEPARATORS: tuple[str, ...] = ("=>", "->", "→", "：", ":", "=")  # noqa: RUF001

# [v3.23.82] Mẫu đánh số tập chuẩn S01E02 / s1e2 (rất ít dương tính giả). Dùng để
# trích tên series từ TÊN FILE - chính xác hơn thư mục cha khi tệp ở gốc ổ đĩa.
_EPISODE_MARKER_RE = re.compile(r"(?i)[._\s-]*s\d{1,2}e\d{1,3}")

__all__ = [
    "SeriesContext",
    "TranslationMemoryEntry",
    "derive_series_key",
    "format_reference_block",
    "merge_characters",
    "merge_glossary",
    "retrieve_relevant",
    "roster_overlap_ratio",
]


@dataclass(frozen=True, slots=True)
class TranslationMemoryEntry:
    """Một cặp câu đã dịch trong bộ nhớ dịch.

    Attributes:
        source_text: Câu nguồn (ngôn ngữ gốc).
        target_text: Câu đích đã dịch (đã được dùng/duyệt ở tập trước).
    """

    source_text: str
    target_text: str


@dataclass(frozen=True, slots=True)
class SeriesContext:
    """Ngữ cảnh chung của một phim bộ, chia sẻ giữa các tập.

    Attributes:
        glossary: Bảng thuật ngữ tích luỹ (mỗi dòng 'gốc => bản dịch').
        characters: Roster nhân vật tích luỹ.
        overview: Tóm tắt cốt truyện chung.
    """

    glossary: str = ""
    characters: str = ""
    overview: str = ""


def merge_glossary(existing: str, incoming: str) -> str:
    """Gộp hai bảng thuật ngữ, khử trùng theo thuật ngữ gốc (giữ mục đã có trước).

    Dùng khi tích luỹ glossary của phim bộ qua các tập: mục mới chỉ được thêm nếu thuật
    ngữ gốc (phần trước dấu phân tách) chưa tồn tại - tránh ghi đè cách dịch đã chốt.

    Args:
        existing: Bảng thuật ngữ hiện có của phim bộ.
        incoming: Bảng thuật ngữ mới (vd từ phân tích tập hiện tại).

    Returns:
        Bảng thuật ngữ đã gộp, mỗi mục một dòng, theo thứ tự: mục cũ trước, mục mới sau.
    """
    def _source_of(line: str) -> str:
        text = line.strip()
        for sep in _SEPARATORS:
            if sep in text:
                return text.split(sep, 1)[0].strip().lower()
        return text.lower()

    merged: list[str] = []
    seen: set[str] = set()
    for block in (existing, incoming):
        for raw in (block or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            src = _source_of(line)
            if src and src not in seen:
                seen.add(src)
                merged.append(line)
    return "\n".join(merged)


def _roster_identity_keys(line: str) -> set[str]:
    """Tập token định danh của một dòng roster (tên đích + alias CJK, đã chuẩn hoá)."""
    keys: set[str] = set()
    match = re.match(r"^(.+?)\s*[(（]\s*([^)）]+?)\s*[)）]", line)  # noqa: RUF001
    if match:
        for part in (match.group(1), match.group(2)):
            for token in re.split(r"[/、,]", part):
                token = token.strip().lower()
                if token:
                    keys.add(token)
    else:
        name = line.split(":", 1)[0].strip().lower()
        if name:
            keys.add(name)
    return keys


def _all_roster_keys(roster: str) -> set[str]:
    keys: set[str] = set()
    for raw in (roster or "").splitlines():
        line = raw.strip()
        if line:
            keys |= _roster_identity_keys(line)
    return keys


def roster_overlap_ratio(roster_a: str, roster_b: str) -> float:
    """Tỉ lệ trùng danh tính giữa hai roster, trong [0, 1].

    Dùng để phát hiện khả năng GỘP NHẦM phim bộ: nếu roster của tập mới gần như không
    trùng roster đã tích luỹ (các tập trước), nhiều khả năng tập này thuộc phim KHÁC bị
    đặt chung thư mục -> nên cảnh báo. Mẫu số là roster nhỏ hơn để nhạy với tập mới.

    Returns:
        0.0 nếu một trong hai roster rỗng.
    """
    keys_a = _all_roster_keys(roster_a)
    keys_b = _all_roster_keys(roster_b)
    if not keys_a or not keys_b:
        return 0.0
    return len(keys_a & keys_b) / min(len(keys_a), len(keys_b))


def merge_characters(existing: str, incoming: str) -> str:
    """Gộp roster nhân vật xuyên tập, khử trùng theo DANH TÍNH (tên chuẩn / alias CJK).

    Giữ mục CŨ khi trùng danh tính (tên đã thiết lập ổn định qua các tập) và thêm nhân
    vật mới ở ``incoming``. Nhờ đó danh sách tên chuẩn TÍCH LUỸ dần, giúp tên riêng nhất
    quán xuyên tập (thay vì mỗi tập một roster độc lập dễ gây trôi tên).

    Hai mục được coi là CÙNG nhân vật nếu CHIA SẺ bất kỳ token định danh nào (tên đích
    hoặc alias CJK, tách theo '/', '、', ',').

    Args:
        existing: Roster đã tích luỹ của phim bộ.
        incoming: Roster mới (từ phân tích tập hiện tại).

    Returns:
        Roster đã gộp, mỗi nhân vật một dòng, mục cũ trước - mục mới sau.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for block in (existing, incoming):
        for raw in (block or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            keys = _roster_identity_keys(line)
            if keys & seen:  # trùng danh tính -> giữ mục đã thêm trước
                continue
            seen |= keys
            merged.append(line)
    return "\n".join(merged)


def retrieve_relevant(
    query_source: str,
    entries: list[TranslationMemoryEntry],
    *,
    top_k: int = 3,
    min_score: float = 70.0,
) -> list[TranslationMemoryEntry]:
    """Trả về tối đa ``top_k`` mục TM giống ``query_source`` nhất (điểm ≥ ``min_score``).

    Dùng ``token_set_ratio`` của rapidfuzz để chịu được khác biệt nhỏ về thứ tự từ/dấu
    câu. Chỉ giữ các mục đạt ngưỡng ``min_score`` (0-100) để tránh tham chiếu nhiễu.

    Args:
        query_source: Câu nguồn cần tìm tham chiếu.
        entries: Toàn bộ mục TM hiện có.
        top_k: Số tham chiếu tối đa trả về.
        min_score: Ngưỡng tương đồng tối thiểu (0-100).

    Returns:
        Danh sách mục TM liên quan, sắp theo độ tương đồng giảm dần. Rỗng nếu không có
        mục nào đạt ngưỡng hoặc đầu vào rỗng.
    """
    query = (query_source or "").strip()
    if not query or not entries or top_k <= 0:
        return []

    # process.extract trả (choice, score, index) — dùng index để lấy lại entry gốc.
    candidates = [entry.source_text for entry in entries]
    matches = process.extract(
        query, candidates, scorer=fuzz.token_set_ratio, limit=top_k
    )
    result: list[TranslationMemoryEntry] = []
    for _matched_text, score, index in matches:
        if score >= min_score:
            result.append(entries[index])
    return result


def format_reference_block(
    entries: list[TranslationMemoryEntry], *, max_entries: int = 8
) -> str:
    """Định dạng các mục TM thành khối tham chiếu để chèn vào prompt hệ thống.

    Trả chuỗi rỗng nếu không có mục nào (để bên gọi quyết định có chèn hay không).

    Args:
        entries: Các mục TM đã truy hồi (đã lọc liên quan).
        max_entries: Số mục tối đa đưa vào khối (tránh prompt quá dài).

    Returns:
        Khối văn bản tham chiếu, hoặc chuỗi rỗng.
    """
    if not entries:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry.source_text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(f'  • "{entry.source_text}" → "{entry.target_text}"')
        if len(lines) >= max_entries:
            break
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "\n- BỘ NHỚ DỊCH (các tập/đoạn trước đã dịch — hãy DÙNG LẠI cách dịch tên riêng, "
        "thuật ngữ và lối xưng hô tương ứng để NHẤT QUÁN xuyên suốt phim bộ):\n" + body
    )


def _series_name_from_filename(file_stem: str) -> str:
    """Trích tên series từ tên tệp nếu có mẫu đánh số tập (SxxExx).

    Ví dụ: ``"NOVA.S51E16.Building.Stuff"`` -> ``"NOVA"``;
    ``"Show.Name.S01E02.720p"`` -> ``"Show.Name"``.

    Args:
        file_stem: Tên tệp đã bỏ phần mở rộng.

    Returns:
        Tên series, hoặc chuỗi rỗng nếu không thấy mẫu / phần trước mẫu rỗng.
    """
    match = _EPISODE_MARKER_RE.search(file_stem)
    if not match:
        return ""
    return file_stem[: match.start()].strip(" ._-")


def derive_series_key(video_path: str | Path) -> str:
    """Suy ra khoá phim bộ từ đường dẫn video, để gom bộ nhớ dịch chung cho các tập.

    Quy ước (theo độ ưu tiên):
      1. Nếu TÊN FILE có mẫu đánh số tập (vd ``NOVA.S51E16``) → lấy phần tên trước mẫu làm
         khoá series. Cách này nhóm đúng các tập cùng phim và TÁCH phim khác nhau, kể
         cả khi tệp ở GỐC ổ đĩa (vd ``G:\\NOVA.S51E16.mkv``) - tránh dùng nhầm chữ cái
         ổ đĩa (``"G:"``) làm khoá khiến mọi phim ở gốc ổ bị gộp chung.
      2. Nếu không có mẫu tập → dùng tên THƯ MỤC CHA (các tập thường nằm chung thư mục).
      3. Không xác định được → trả chuỗi rỗng (bên gọi sẽ bỏ qua TM).

    Args:
        video_path: Đường dẫn tệp video, dạng ``str`` hoặc ``pathlib.Path`` (có thể rỗng).
            Chấp nhận cả hai kiểu vì bên gọi có thể truyền ``WindowsPath`` từ tầng UI.

    Returns:
        Khoá series, hoặc chuỗi rỗng.
    """
    # Chuẩn hoá về chuỗi: hỗ trợ cả ``Path`` (``WindowsPath``/``PosixPath``) lẫn ``str``.
    # Tránh ``str(None)`` -> "None" bằng cách kiểm tra falsy trước khi ép kiểu.
    path = (str(video_path) if video_path else "").strip()
    if not path:
        return ""
    # Tách thủ công để không phụ thuộc hệ điều hành (xử lý cả "/" và "\\").
    normalized = path.replace("\\", "/").rstrip("/")
    segments = [seg for seg in normalized.split("/") if seg]
    if not segments:
        return ""

    # Ưu tiên 1: trích series từ tên file (mẫu SxxExx).
    file_name = segments[-1]
    file_stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    series_from_name = _series_name_from_filename(file_stem)
    if series_from_name:
        return series_from_name

    # Ưu tiên 2: tên thư mục cha (phần tử áp cuối).
    if len(segments) < 2:
        return ""
    return segments[-2].strip()
