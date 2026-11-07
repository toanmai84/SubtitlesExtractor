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
    "episode_label",
    "format_reference_block",
    "merge_characters",
    "merge_glossary",
    "recent_series_overview",
    "retrieve_for_lines",
    "retrieve_relevant",
    "roster_overlap_ratio",
    "split_episode_overviews",
    "upsert_episode_overview",
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
        # [v3.23.669] Cắt cả ở " — " / " - " (dòng "Tên — vai trò", vd người nói bổ sung): trước đây
        # cả dòng thành khoá, nên thêm mệnh đề vào cuối dòng là đổi danh tính và sinh dòng trùng.
        name = re.split(r":|\s[—–-]\s", line, maxsplit=1)[0].strip().lower()
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
    resolve = _partner_identity_resolver(existing, incoming)   # dựng MỘT lần (O(R) thay vì O(R²))
    for block in (existing, incoming):
        for raw in (block or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            keys = _roster_identity_keys(line)
            if keys & seen:
                # [v3.23.669] Trùng danh tính → giữ dòng đã có (tên, phiên âm, mô tả đã chốt) nhưng
                # BỔ SUNG mệnh đề xưng hô với người MỚI. Đo 8 cặp tập 1+2 phim mẫu: giữ nguyên dòng
                # cũ làm mất quy tắc của tập 2 ở 7/8 lần ("với Nam Cung Mộng tự xưng 'đệ tử' gọi
                # 'sư thúc'" — Nam Cung Mộng chỉ xuất hiện từ tập 2); gộp mệnh đề: 134 → 139/166.
                index = next(
                    (i for i, kept in enumerate(merged) if _roster_identity_keys(kept) & keys),
                    None,
                )
                if index is not None:
                    merged[index] = _with_new_address_clauses(merged[index], line, resolve)
                continue
            seen |= keys
            merged.append(line)
    return "\n".join(merged)


#: [v3.23.669] Một mệnh đề xưng hô của bảng nhân vật: "với X tự xưng 'a' gọi 'b' (danh xưng: …)".
_ADDRESS_CLAUSE_RE = re.compile(
    r"(?:^|(?<=[;:—–,]))\s*(?:xưng hô:\s*)?"
    r"(?P<clause>với\s+(?P<who>[^;,.:]+?)\s+tự xưng\s+['‘“\"][^'’”\"]*['’”\"]\s+gọi\s+"
    r"['‘“\"][^'’”\"]*['’”\"](?:\s*\([^)]*\))?)",
    re.MULTILINE,
)
#: Kính ngữ / đại từ hay đứng trước tên người đối diện ("nàng Sở Dao").
_PARTNER_PREFIXES = ("nàng ", "chàng ", "cô ", "anh ", "chị ", "ông ", "bà ", "lão ", "tiểu ")


def _partner_identity_resolver(*rosters: str):
    """Hàm ``tên người đối diện → khoá danh tính`` theo các bảng nhân vật (bí danh cùng một người)."""
    groups: list[set[str]] = []
    for roster in rosters:
        for raw in (roster or "").splitlines():
            keys = _roster_identity_keys(raw.strip()) if raw.strip() else set()
            if not keys:
                continue
            for group in groups:
                if group & keys:
                    group |= keys
                    break
            else:
                groups.append(set(keys))

    def resolve(name: str) -> str:
        key = re.sub(r"\s*[(（][^)）]*[)）]", "", name).strip().lower()
        candidates = [key]
        for prefix in _PARTNER_PREFIXES:
            if key.startswith(prefix):
                candidates.append(key[len(prefix):].strip())
        for candidate in candidates:
            for index, group in enumerate(groups):
                if candidate in group:
                    return f"#{index}"
        return key

    return resolve


def _with_new_address_clauses(kept: str, incoming: str, resolve=lambda name: name.strip().lower()) -> str:
    """Thêm vào ``kept`` các mệnh đề xưng hô của ``incoming`` với người ``kept`` chưa nhắc tới.

    ``resolve`` quy tên người đối diện về danh tính ("Sở Dao" và "Dao Dao" là một người) để không thêm
    mệnh đề thứ hai mâu thuẫn cho cùng một cặp.
    """
    have = {resolve(m.group("who")) for m in _ADDRESS_CLAUSE_RE.finditer(kept)}
    extra: list[str] = []
    for match in _ADDRESS_CLAUSE_RE.finditer(incoming):
        who = resolve(match.group("who"))
        if who not in have:
            have.add(who)
            extra.append(match.group("clause"))
    if not extra:
        return kept
    base = kept.rstrip(" ;.")
    # Dòng chỉ có tên (không ":" / "(" / "—"): nối bằng ": " để khoá danh tính của dòng không đổi.
    joiner = "; " if re.search(r"[:(（—–]|\s-\s", base) else ": "
    return base + joiner + "; ".join(extra)


#: [v3.23.669] Trần độ dài tóm tắt phim bộ tích luỹ (~3k token) — gửi kèm khi phân tích tập sau.
MAX_SERIES_OVERVIEW_CHARS = 12_000
#: Đầu khối của một tập: "⟦nhãn⟧ " ở đầu dòng. Dấu ⟦⟧ (không dùng "[…]") để tóm tắt của model có dòng
#: "[Cảnh 1] …" / "[Hồi tưởng] …" không bị hiểu nhầm là ranh giới tập.
_EPISODE_BLOCK_RE = re.compile(r"^⟦([^⟧\n]{1,80})⟧\s?", re.MULTILINE)
_LABEL_MAX = 80


def episode_label(stem: str) -> str:
    """[v3.23.669] Nhãn tập dùng chung cho ghi và tra (tên tệp video, bỏ ký tự đặc biệt).

    Tên dài hơn trần thì giữ PHẦN CUỐI — số tập thường nằm cuối ("… Episode 01").
    """
    label = re.sub(r"[⟦⟧\r\n]+", " ", stem or "").strip()
    return label if len(label) <= _LABEL_MAX else label[-_LABEL_MAX:]


def split_episode_overviews(series_overview: str) -> list[tuple[str, str]]:
    """Tách tóm tắt phim bộ thành ``[(nhãn tập, tóm tắt), …]`` theo thứ tự (hàm thuần).

    Văn bản cũ không có nhãn (bản trước chỉ giữ tóm tắt của tập dịch gần nhất) thành một khối nhãn ``""``.
    """
    text = (series_overview or "").strip()
    if not text:
        return []
    marks = list(_EPISODE_BLOCK_RE.finditer(text))
    if not marks:
        return [("", text)]
    blocks: list[tuple[str, str]] = []
    head = text[: marks[0].start()].strip()
    if head:
        blocks.append(("", head))
    for k, mark in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        blocks.append((mark.group(1).strip(), text[mark.end():end].strip()))
    return blocks


def upsert_episode_overview(
    series_overview: str, episode: str, overview: str,
    max_chars: int = MAX_SERIES_OVERVIEW_CHARS,
) -> str:
    """[v3.23.669] Ghi tóm tắt của MỘT tập vào tóm tắt phim bộ (thay khối cũ của chính tập đó).

    Trước đây mỗi tập ghi đè tóm tắt của tập trước, nên phân tích tập N không biết diễn biến các tập
    trước, và "Dịch cả bộ" dùng tóm tắt của tập khác cho tập chưa phân tích. Nay giữ mỗi tập một khối
    "[tên tập] …" theo thứ tự; vượt trần thì bỏ các khối cũ ở GIỮA (giữ tập đầu = thiết lập bối cảnh
    và các tập gần nhất).

    Args:
        series_overview: Tóm tắt phim bộ đang lưu.
        episode: Nhãn tập (vd tên tệp video ``"01"``).
        overview: Tóm tắt của tập (rỗng = không đổi).
        max_chars: Trần độ dài.

    Returns:
        Tóm tắt phim bộ mới.
    """
    episode = episode_label(episode)
    body = (overview or "").strip()
    existing = split_episode_overviews(series_overview)
    # Tóm tắt đã mang nhãn tập, hoặc trùng nguyên văn một khối đã có / cả tóm tắt phim bộ (ô tóm tắt
    # được điền từ chính tóm tắt phim bộ khi tập chưa phân tích) → không phải của riêng tập này.
    if (
        not body or _EPISODE_BLOCK_RE.search(body)
        or any(text == body for label, text in existing if label != episode)
        or body == (series_overview or "").strip()
    ):
        return (series_overview or "").strip()
    blocks = [(label, text) for label, text in existing if label != episode]
    blocks.append((episode, body))
    while len(blocks) > 2 and len(_render_blocks(blocks)) > max_chars:
        del blocks[1]                                   # giữ tập đầu + các tập mới nhất
    if len(_render_blocks(blocks)) > max_chars:
        # Còn ≤ 2 khối mà vẫn quá dài: rút gọn PHẦN THÂN (giữ nguyên nhãn) thay vì cắt chuỗi.
        budget = max(200, max_chars // len(blocks) - _LABEL_MAX - 4)
        blocks = [(label, text if len(text) <= budget else text[:budget].rstrip() + " …")
                  for label, text in blocks]
    return _render_blocks(blocks)


def _render_blocks(items: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"⟦{label}⟧ {text}" if label else text for label, text in items)


def recent_series_overview(series_overview: str, exclude_episode: str = "",
                           max_chars: int = 6000) -> str:
    """[v3.23.669] Diễn biến các tập TRƯỚC (bỏ khối của chính tập đang phân tích), gọn ``max_chars``."""
    exclude = episode_label(exclude_episode) if exclude_episode else ""
    blocks = [(label, text) for label, text in split_episode_overviews(series_overview)
              if not exclude or label != exclude]
    while len(blocks) > 1 and len(_render_blocks(blocks)) > max_chars:
        del blocks[0]                                   # gửi kèm các tập GẦN NHẤT
    out = _render_blocks(blocks)
    return out if len(out) <= max_chars else "…" + out[-max_chars:]


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


_QUERY_CHUNK = 50


def retrieve_for_lines(
    queries: list[str],
    entries: list[TranslationMemoryEntry],
    *,
    per_line_k: int = 2,
    max_references: int = 40,
    min_score: float = 70.0,
) -> list[TranslationMemoryEntry]:
    """[v3.23.654] Truy hồi tham chiếu cho NHIỀU câu một lượt (``rapidfuzz.process.cdist``, đa luồng).

    Thay vòng ``retrieve_relevant`` từng câu (400 câu × mọi mục, đo 4–7 s trên luồng giao diện với phim 3648 dòng).
    Thứ tự kết quả giữ như cũ: theo thứ tự ``queries``, mỗi câu tối đa ``per_line_k`` mục điểm cao nhất, bỏ trùng,
    dừng ở ``max_references``.
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries or not entries or per_line_k <= 0:
        return []
    import numpy as np

    candidates = [entry.source_text for entry in entries]
    collected: list[TranslationMemoryEntry] = []
    seen: set[str] = set()
    # Từng khúc nhỏ: đủ tham chiếu thì dừng sớm như vòng cũ (dữ liệu khớp nhiều), mỗi khúc chạy song song.
    for start in range(0, len(queries), _QUERY_CHUNK):
        scores = process.cdist(
            queries[start:start + _QUERY_CHUNK], candidates,
            scorer=fuzz.token_set_ratio, score_cutoff=min_score, workers=-1,
        )
        for row in scores:
            hits = np.flatnonzero(row >= min_score)
            if hits.size == 0:
                continue
            # Điểm giảm dần; cùng điểm thì mục đứng trước (mới hơn) thắng — như process.extract.
            order = hits[np.lexsort((hits, -row[hits]))][:per_line_k]
            for index in order.tolist():
                source = entries[index].source_text
                if source not in seen:
                    seen.add(source)
                    collected.append(entries[index])
            if len(collected) >= max_references:
                return collected[:max_references]
    return collected[:max_references]


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


#: [v3.23.564] Thư mục cha là gốc ổ đĩa Windows ("D:") — không phải tên phim bộ.
_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:$")


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
    parent = segments[-2].strip()
    # [v3.23.564] Tệp nằm ở GỐC ổ đĩa (``D:\phim.mp4``) và không có mẫu số tập: thư mục
    # cha là CHỮ CÁI Ổ ĐĨA, không phải tên phim bộ. Trước đây trả "D:" → mọi phim lẻ ở
    # gốc ổ bị gộp làm một "phim bộ": dùng chung bộ nhớ dịch, và phân tích lại CHÍNH phim
    # đó bị ép "tái sử dụng tên đã chốt ở các tập trước" — tức bị neo vào bản phân tích
    # cũ (ca thật: phân tích lại bằng gemini-3.8-flash bị buộc dùng lại roster 3.6-flash).
    if _DRIVE_ROOT_RE.match(parent):
        return ""
    return parent
