"""[v3.23.581] Nhãn Visual Cues của bộ QUÉT BỔ SUNG mâu thuẫn với cuộc hội thoại quanh nó.

Chuẩn đoán thật (phim "he thong sung the", 5540 dòng, phân tích lại ở v3.23.578): cues
của khâu phân tích chỉ phủ 57% nên khâu dịch quét bổ sung 2358 dòng. Nhãn quét bổ sung
(nhận ra nhờ luôn có mô tả bối cảnh) sai người nghe ở 15/51 dòng GỌI TÊN (29%), nhãn của
khâu phân tích chỉ sai 9/131 (7%). Sai theo kiểu "lạc cảnh": giữa cuộc gọi của Lâm Dật
với chị dâu (dòng 1303 và 1308 đều là cặp 林逸 ↔ 宋欣爽), các dòng 1304-1307 "你在医院吗 /
我有个好消息 / 要告诉你 / 我来找你" lại mang nhãn 付金奎 → 王大福 kèm mô tả "Phó Kim Khuê
hét lên đe dọa" — một cảnh của đoạn phim sau. Nhãn sai đi thẳng vào khâu dịch (quy tắc
xưng hô của cặp sai được gắn vào dòng) và vào mọi số liệu xưng hô của file chuẩn đoán.

Tín hiệu: dòng quét bổ sung nằm giữa hai nhãn tin cậy CÙNG một cặp nhân vật, cách nhau ít
dòng và ít giây, mà cặp của nó không có ai chung với cặp đó — trừ khi chính câu gốc gọi
tên người nghe theo nhãn quét (người thứ ba chen vào: "阿秀你沒事吧" giữa cuộc đối đầu
của hai người khác). Trên phim này: 97 dòng; soát tay 25 dòng lấy ngẫu nhiên, gần như tất
cả là nhãn lạc cảnh. Ở các phim mà cues chỉ đến từ khâu phân tích, không có nhãn quét
nào nên không dòng nào bị đụng tới.

Toàn bộ là hàm thuần: không I/O.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable

from subtitles_extractor.application.services.cue_run_filler import LineSpan
from subtitles_extractor.application.services.roster_format import name_aliases
from subtitles_extractor.application.services.vocative_guard import vocative_alias_index
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue
from subtitles_extractor.domain.value_objects.speaker_label import is_fallback_speaker

#: Hai nhãn tin cậy bao quanh cách nhau tối đa ngần này số dòng.
CONTRADICTION_MAX_LINES = 6
#: ... và tối đa ngần này mili-giây (từ lúc dòng trước kết thúc tới lúc dòng sau bắt đầu).
CONTRADICTION_MAX_GAP_MS = 12_000
#: Phiên cũ không ghi nguồn gốc nhãn: chỉ coi nhãn có mô tả là nhãn quét khi nhãn KHÔNG
#: mô tả (của khâu phân tích) chiếm ít nhất tỉ lệ này — nếu không, cả tệp là một nguồn.
MIN_TRUSTED_SHARE = 0.3


def _people(label: str) -> frozenset[str]:
    return frozenset(name_aliases(label))


def _pair(cue: VisualCue) -> tuple[frozenset[str], ...]:
    """Cặp người nói/người nghe (không phân biệt chiều), mỗi người là tập bí danh."""
    sides = {_people(label) for label in (cue.speaker, cue.addressee) if label}
    return tuple(sorted(sides, key=sorted))


def _everyone(cue: VisualCue) -> frozenset[str]:
    return _people(cue.speaker) | _people(cue.addressee)


def described_scan_lines(cues: Iterable[VisualCue]) -> set[int]:
    """Dòng mang nhãn của bộ quét (có mô tả bối cảnh) trong phiên không ghi nguồn gốc.

    Args:
        cues: Visual Cues đã lưu.

    Returns:
        Tập số dòng; rỗng khi nhãn không mô tả chiếm dưới :data:`MIN_TRUSTED_SHARE`.
    """
    labelled = [cue for cue in cues if cue.speaker or cue.addressee]
    if not labelled:
        return set()
    described = {cue.line_no for cue in labelled if cue.scene.strip()}
    plain_share = (len(labelled) - len(described)) / len(labelled)
    return described if plain_share >= MIN_TRUSTED_SHARE else set()


def _mentioned_names(characters: str) -> dict[str, frozenset[str]]:
    """``bí danh nhãn → mọi bí danh chữ Hán của người đó`` để nhận câu gọi tên."""
    index = vocative_alias_index(characters)
    by_entry: dict[int, set[str]] = {}
    for alias, entry in index.items():
        by_entry.setdefault(id(entry), set()).add(alias)
    return {
        alias: frozenset(by_entry[id(entry)]) for alias, entry in index.items()
    }


def _names_listener(
    cue: VisualCue, source: str, mentioned: dict[str, frozenset[str]]
) -> bool:
    """Câu gốc gọi tên chính người nghe theo nhãn quét ("阿秀你沒事吧" → 秦可秀)."""
    aliases = {
        name for label in _people(cue.addressee) for name in mentioned.get(label, ())
    }
    return any(alias in source for alias in aliases)


def contradicted_cue_lines(
    cues: Iterable[VisualCue],
    untrusted: set[int],
    spans: dict[int, LineSpan],
    sources: dict[int, str] | None = None,
    characters: str = "",
) -> list[int]:
    """Nhãn không tin cậy mâu thuẫn với cặp nhân vật của hai nhãn tin cậy bao quanh.

    Args:
        cues: Visual Cues (đã quy nhãn về tên chuẩn của bảng nhân vật).
        untrusted: Số dòng mang nhãn không tin cậy (nhãn quét bổ sung).
        spans: ``số dòng → thời gian`` của các dòng phụ đề.
        sources: ``số dòng → câu gốc`` — câu gọi tên người nghe theo nhãn quét thì giữ.
        characters: Bảng nhân vật (để nhận tên trong câu gốc).

    Returns:
        Số dòng cần bỏ nhãn, tăng dần.
    """
    mentioned = _mentioned_names(characters) if sources and characters else {}
    by_line = {cue.line_no: cue for cue in cues}
    trusted = sorted(
        line_no
        for line_no, cue in by_line.items()
        if line_no not in untrusted and len(_pair(cue)) == 2
    )
    contradicted: list[int] = []
    for line_no in sorted(untrusted):
        cue = by_line.get(line_no)
        if cue is None or not _everyone(cue):
            continue
        # [v3.23.600] Nhãn dự phòng chỉ ghi GIỌNG ("Giọng nam chưa rõ"), không khẳng định
        # danh tính nên không thể "lạc cảnh"; bỏ nó là trả dòng về không có người nói.
        if is_fallback_speaker(cue.speaker) and not cue.addressee:
            continue
        position = bisect.bisect_left(trusted, line_no)
        if position == 0 or position >= len(trusted):
            continue
        before, after = trusted[position - 1], trusted[position]
        if after - before > CONTRADICTION_MAX_LINES:
            continue
        if before not in spans or after not in spans:
            continue
        if spans[after].start_ms - spans[before].end_ms > CONTRADICTION_MAX_GAP_MS:
            continue
        surrounding = _pair(by_line[before])
        if surrounding != _pair(by_line[after]):
            continue
        if _everyone(cue) & _everyone(by_line[before]):
            continue
        source = (sources or {}).get(line_no, "")
        if mentioned and _names_listener(cue, source, mentioned):
            continue
        contradicted.append(line_no)
    return contradicted


__all__ = [
    "CONTRADICTION_MAX_GAP_MS",
    "CONTRADICTION_MAX_LINES",
    "MIN_TRUSTED_SHARE",
    "contradicted_cue_lines",
    "described_scan_lines",
]
