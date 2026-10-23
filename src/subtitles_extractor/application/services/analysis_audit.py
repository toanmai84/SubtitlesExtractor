"""[v3.23.624] Tự kiểm tra & vá khiếm khuyết của kết quả phân tích ngữ cảnh (hàm thuần).

Người dùng soát kết quả phân tích mà thấy chưa đúng thì KHÔNG nên bắt phân tích lại từ đầu
(tốn quota, lại ra kết quả ngẫu nhiên khác). Mô-đun này lấy chính phân tích cũ làm gốc:

1. :func:`find_analysis_defects` đối chiếu tất định bảng nhân vật / thuật ngữ / tóm tắt / nhãn
   người nói với phụ đề nguồn, liệt kê khiếm khuyết cụ thể (không tốn lời gọi AI).
2. Nếu có khiếm khuyết, một lời gọi nhỏ hỏi model CHỈ phần cần sửa (bản vá) — xem
   ``audit_analysis`` ở adapter.
3. :func:`apply_audit_patch` gộp bản vá vào phân tích cũ; mục không bị nhắc tới giữ nguyên.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonicalize_cue_labels,
    is_roster_label,
    roster_label_map,
)
from subtitles_extractor.application.services.roster_format import (
    RosterEntry,
    address_clauses,
    lacks_address_rules,
    parse_glossary,
    parse_roster,
    render_glossary,
    render_roster,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
    serialize_visual_cues,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import (
    SubtitleContextAnalysis,
    VisualCue,
)
from subtitles_extractor.domain.value_objects.speaker_label import (
    VOICE_ONLY_SPEAKERS,
    canonical_fallback_speaker,
)

#: Nhãn ngoài bảng nhân vật xuất hiện từ chừng này dòng thì coi là nhân vật bị sót.
UNLISTED_LABEL_MIN_LINES = 3
#: Trần số khiếm khuyết gửi cho model trong một lượt (giữ prompt nhỏ).
MAX_DEFECTS = 60
#: Số lượt tự kiểm tra tối đa (lượt sau chỉ chạy khi lượt trước còn khiếm khuyết VÀ có thay đổi).
MAX_AUDIT_ROUNDS = 2
#: [v3.23.637] Bảng nhân vật không có MỘT mệnh đề xưng hô nào đọc được (ca thật: phim 86 tập,
#: 19 nhân vật, 0 mệnh đề; tái hiện 1/6 lượt phân tích phim ghép 12 tập với gemini-3.8-flash).
NO_ADDRESS_RULES_KIND = "no_address_rules"
#: Loại khiếm khuyết chỉ mang tính gợi ý, không tự nó đáng một lời gọi AI.
STALE_GLOSSARY_KIND = "stale_glossary"

# ── [v3.23.631] Tự kiểm tra RẺ + HỘI TỤ ───────────────────────────────────────────────────
# Nhật ký thật (phim 86 tập, 3649 dòng, gemini-3.8-flash, 20 request/ngày/key): lượt 1 mất
# 10 phút — thinking "high" tốn 16-30 nghìn token suy nghĩ, một lời gọi chạy hết 300 giây thì
# 504 (cả hai lần thử lại cũng 504, mỗi lần bị tính một request), vì 40 dòng nghi lỗi × cửa sổ
# ±40 dòng gần như phủ CẢ PHIM vào một prompt. Và các khiếm khuyết model không sửa được
# (self_address hợp lệ như độc thoại, nhãn cố tình ngoài bảng…) lại được gửi đi ở MỌI lần
# chạy: lần sau vẫn 17 khiếm khuyết như lần trước.
#: Model kiểm định mặc định: bản vá là việc cơ học, hạn mức 500 request/ngày — không tiêu hạn
#: mức 20 request/ngày của model phân tích tốt nhất (mà 24 đoạn video đã cần).
DEFAULT_AUDIT_MODEL = "gemini-3.1-flash-lite"
#: Mức suy nghĩ của bước kiểm định. Đo thật (Gemini 3.1 Flash-Lite, tập 2 bị cố ý làm hỏng 12
#: khiếm khuyết, 3 lượt mỗi mức): "medium" sạch 3/3 lượt trong 3,8-5,2 giây (550-880 token suy nghĩ);
#: "low" chỉ 1/3 lượt sạch và vá thừa tới 39 mục (hỏng thêm); "high" 2/3 lượt sạch, chậm gấp 2-5 lần.
#: (Mức "high" của model phân tích mới là thứ gây 16-30 nghìn token suy nghĩ + 504 trong log thật.)
AUDIT_THINKING_LEVEL = "medium"
#: Cửa sổ dòng quanh mỗi dòng nghi lỗi gửi cho model (đủ thấy lượt thoại kề để đoán người nói).
AUDIT_WINDOW_LINES = 8
#: Trần số dòng nguồn trong MỘT lời gọi kiểm định (prompt nhỏ → nhanh, không chạm hạn chót).
AUDIT_MAX_PROMPT_LINES = 320
#: Số lời gọi kiểm định tối đa cho MỘT lần tự kiểm tra (phần còn lại để lần sau).
AUDIT_MAX_CALLS = 6
#: Khiếm khuyết bị model bỏ qua (gửi mà vẫn còn) quá ngần này lần thì thôi không gửi nữa.
DECLINE_LIMIT = 3


@dataclass(frozen=True, slots=True)
class AnalysisDefect:
    """Một khiếm khuyết tìm được.

    Attributes:
        kind: ``empty_roster`` | ``empty_overview`` | ``unlisted_label`` | ``missing_speaker`` |
            ``self_address`` | ``stale_glossary``.
        detail: Mô tả cụ thể cho model/nhật ký.
        line_ids: Các dòng liên quan (rỗng nếu lỗi ở bảng).
    """

    kind: str
    detail: str
    line_ids: tuple[int, ...] = ()


def _line_text(line: Any) -> str:
    return str(getattr(line, "original_text", None) or getattr(line, "text", "") or "")


def find_analysis_defects(
    analysis: SubtitleContextAnalysis, source_lines: list[Any]
) -> list[AnalysisDefect]:
    """Đối chiếu phân tích với phụ đề nguồn, trả danh sách khiếm khuyết (tối đa ``MAX_DEFECTS``)."""
    spoken = [ln for ln in source_lines if _line_text(ln).strip()]
    if not spoken:
        return []
    defects: list[AnalysisDefect] = []
    roster = parse_roster(analysis.characters)
    if not roster:
        defects.append(AnalysisDefect("empty_roster", "Bảng nhân vật rỗng."))
    if not (analysis.overview or "").strip():
        defects.append(AnalysisDefect("empty_overview", "Tóm tắt cốt truyện rỗng."))
    if roster and lacks_address_rules(analysis.characters):
        names = "; ".join(e.name.split("/")[0].strip() for e in roster)
        defects.append(AnalysisDefect(
            NO_ADDRESS_RULES_KIND,
            "Bảng nhân vật KHÔNG có mệnh đề xưng hô đọc được. Với MỖI nhân vật có thoại, viết lại "
            "dòng của họ (giữ tên gốc, giới, tuổi, vai trò) và thêm ĐÚNG dạng: xưng hô: với <Tên "
            "người nghe, đúng như bảng> tự xưng '<một đại từ>' gọi '<một đại từ>' (mỗi người nghe "
            "một mệnh đề, cách nhau dấu phẩy). Tên trong bảng: " + names + "."))

    full_text = "\n".join(_line_text(ln) for ln in spoken)
    for entry in parse_glossary(analysis.glossary):
        term = entry.term.strip()
        if len(term) >= 2 and entry.translation.strip() and term not in full_text:
            defects.append(AnalysisDefect(
                STALE_GLOSSARY_KIND, f"Thuật ngữ '{term}' không xuất hiện trong phụ đề tập này."))

    cues = {c.line_no: c for c in deserialize_visual_cues(analysis.visual_cues)}
    if cues:
        label_map = roster_label_map(analysis.characters)
        unlisted: Counter[str] = Counter()
        missing: list[int] = []
        selfs: list[int] = []
        for line in spoken:
            cue = cues.get(line.index)
            if cue is None or not cue.speaker.strip():
                missing.append(line.index)
                continue
            if cue.speaker.strip() and cue.speaker.strip() == cue.addressee.strip():
                selfs.append(line.index)
            for label in (cue.speaker, cue.addressee):
                label = label.strip()
                if (
                    label and label not in VOICE_ONLY_SPEAKERS
                    and not is_roster_label(label, label_map)
                    and not canonical_fallback_speaker(label.strip().lower())
                ):
                    unlisted[label] += 1
        for label, count in unlisted.most_common():
            if count >= UNLISTED_LABEL_MIN_LINES:
                ids = tuple(
                    ln.index for ln in spoken
                    if ln.index in cues
                    and label in (cues[ln.index].speaker, cues[ln.index].addressee)
                )
                defects.append(AnalysisDefect(
                    "unlisted_label",
                    f"'{label}' xuất hiện ở {count} dòng nhưng không có trong bảng nhân vật "
                    "(thiếu nhân vật, hoặc viết sai/khác tên so với bảng).", ids[:12]))
        if missing:
            defects.append(AnalysisDefect(
                "missing_speaker", f"{len(missing)} dòng có lời nhưng chưa có người nói.",
                tuple(missing[:40])))
        if selfs:
            defects.append(AnalysisDefect(
                "self_address", f"{len(selfs)} dòng ghi người nghe trùng người nói (nghi sai).",
                tuple(selfs[:40])))
    return defects[:MAX_DEFECTS]


def _entry_key(entry: RosterEntry) -> str:
    return entry.name.split("/")[0].strip().lower()


def _has_address_clause(entry: RosterEntry) -> bool:
    return bool(address_clauses(render_roster([entry])))


def _address_tail(role: str) -> str:
    """Phần "xưng hô: với … tự xưng '…' gọi '…'" của một vai trò (rỗng nếu không có)."""
    at = role.find("xưng hô:")
    if at < 0:
        match = re.search(r"với\s+\S+.*?xưng\s+'", role)
        at = match.start() if match else -1
    return role[at:].strip() if at >= 0 else ""


def _role_with_address(old: RosterEntry, new: RosterEntry) -> str:
    """[v3.23.637] Vai trò sau khi gộp: giữ vai trò cũ (kể cả bản người dùng sửa); CHỈ khi nhân vật
    chưa có mệnh đề xưng hô nào mà bản vá mang mệnh đề đọc được thì nối phần xưng hô đó vào."""
    if not old.role:
        return new.role
    if _has_address_clause(old):
        return old.role
    tail = _address_tail(new.role)
    if not tail or not _has_address_clause(replace(new, role=tail)):
        return old.role
    return f"{old.role.rstrip(' .;,')}; {tail if tail.startswith('xưng hô') else 'xưng hô: ' + tail}"


def _items(value: Any) -> list[Any]:
    """[v3.23.638] Các phần tử CHUỖI của một trường bản vá; trường sai kiểu (số, chuỗi đơn, null…) và
    phần tử không phải chuỗi (``None`` → "None" thành nhân vật giả) bị bỏ."""
    return [x for x in value if isinstance(x, str)] if isinstance(value, (list, tuple)) else []


def apply_audit_patch(
    analysis: SubtitleContextAnalysis, patch: dict[str, Any], valid_line_ids: set[int],
    source_text: str = "",
) -> tuple[SubtitleContextAnalysis, int]:
    """Gộp bản vá của model vào phân tích cũ.

    Args:
        analysis: Phân tích hiện tại.
        patch: ``characters`` (các dòng nhân vật thêm/sửa), ``remove_characters`` (tên),
            ``glossary`` (dòng ``gốc = dịch`` thêm/sửa), ``remove_glossary`` (thuật ngữ gốc),
            ``overview`` (văn bản mới; rỗng = giữ), ``cue_fixes`` (``[{id, spk, to}]``).
        valid_line_ids: Số dòng hợp lệ (cue_fixes ngoài tập này bị bỏ).
        source_text: Toàn văn phụ đề; có thì chỉ cho xoá thuật ngữ KHÔNG xuất hiện trong đó.

    Chỉnh sửa tay của người dùng được bảo vệ: nhân vật đã có chỉ được ĐIỀN phần còn trống (không
    ghi đè vai trò/xưng hô); không xoá nhân vật còn được nhãn người nói nhắc tới.

    Returns:
        ``(phân tích mới, số thay đổi)``. Mục không được nhắc tới giữ nguyên.
    """
    if not isinstance(patch, Mapping):
        return analysis, 0
    changes = 0
    characters = analysis.characters
    entries = parse_roster(analysis.characters)
    by_key = {_entry_key(e): i for i, e in enumerate(entries)}
    used = {
        label.strip().lower()
        for cue in deserialize_visual_cues(analysis.visual_cues)
        for label in (cue.speaker, cue.addressee) if label.strip()
    }
    removed = {
        str(n).strip().lower() for n in _items(patch.get("remove_characters"))
        if str(n).strip() and str(n).strip().lower() not in used
    }
    for raw in _items(patch.get("characters")):
        for new in parse_roster(str(raw)):
            key = _entry_key(new)
            if not key:
                continue
            idx = by_key.get(key)
            if idx is None:
                entries.append(new)
                by_key[key] = len(entries) - 1
                changes += 1
                continue
            old = entries[idx]
            merged = RosterEntry(
                name=old.name or new.name, aka=old.aka or new.aka,
                gender=old.gender or new.gender, age=old.age or new.age,
                role=_role_with_address(old, new),
            )
            if merged != old:
                entries[idx] = merged
                changes += 1
    kept = [e for e in entries if _entry_key(e) not in removed]
    changes += len(entries) - len(kept)
    if changes:
        characters = render_roster(kept)

    glossary = analysis.glossary
    gentries = parse_glossary(glossary)
    gindex = {e.term.strip().lower(): i for i, e in enumerate(gentries)}
    gchanged = 0
    for new in parse_glossary("\n".join(str(x) for x in _items(patch.get("glossary")))):
        idx = gindex.get(new.term.strip().lower())
        if idx is None:
            gentries.append(new)
            gindex[new.term.strip().lower()] = len(gentries) - 1
            gchanged += 1
        elif new.translation and gentries[idx].translation != new.translation:
            gentries[idx] = new
            gchanged += 1
    gremoved = {
        str(t).strip().lower() for t in _items(patch.get("remove_glossary"))
        if str(t).strip() and (not source_text or str(t).strip() not in source_text)
    }
    gkept = [e for e in gentries if e.term.strip().lower() not in gremoved]
    gchanged += len(gentries) - len(gkept)
    if gchanged:
        glossary = render_glossary(gkept)
        changes += gchanged

    overview = analysis.overview
    new_overview = (
        patch.get("overview").strip() if isinstance(patch.get("overview"), str) else ""
    )
    if new_overview and new_overview != (overview or "").strip():
        overview = new_overview
        changes += 1

    visual_cues = analysis.visual_cues
    fixes = patch.get("cue_fixes") if isinstance(patch.get("cue_fixes"), (list, tuple)) else []
    if fixes and valid_line_ids:
        cues = {c.line_no: c for c in deserialize_visual_cues(visual_cues)}
        fixed: list[VisualCue] = []
        for item in fixes:
            if not isinstance(item, dict):
                continue
            try:
                line_no = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            speaker = str(item.get("spk") or "").strip()
            if line_no not in valid_line_ids or not speaker:
                continue
            old = cues.get(line_no) or VisualCue(line_no=line_no)
            new = replace(
                old, speaker=speaker,
                addressee=str(item.get("to") or "").strip() or old.addressee,
                scene=old.scene or str(item.get("cue") or "").strip(),
            )
            fixed.append(new)
        if fixed:
            fixed = canonicalize_cue_labels(fixed, characters)
            for cue in fixed:
                if cues.get(cue.line_no) != cue:
                    cues[cue.line_no] = cue
                    changes += 1
            visual_cues = serialize_visual_cues([cues[k] for k in sorted(cues)])
    return replace(
        analysis, characters=characters, glossary=glossary, overview=overview,
        visual_cues=visual_cues,
    ), changes


@dataclass(slots=True)
class AuditReport:
    """Tóm tắt một lần tự kiểm tra (để giao diện báo người dùng).

    Attributes:
        found: Số khiếm khuyết tìm thấy ở lượt đầu (gồm gợi ý thuật ngữ).
        blocked: Số khiếm khuyết bị bỏ qua vì model đã không sửa được ``DECLINE_LIMIT`` lần.
        calls: Số lời gọi AI đã chạy.
        patched: Số mục đã vá.
        remaining: Số khiếm khuyết (không tính gợi ý) còn lại và còn đáng gửi.
    """

    found: int = 0
    blocked: int = 0
    calls: int = 0
    patched: int = 0
    remaining: int = 0


def describe_audit_report(report: AuditReport | None) -> str:
    """Câu tiếng Việt tóm tắt một lần tự kiểm tra (rỗng nếu không có số liệu)."""
    if report is None:
        return ""
    if report.found == 0:
        return "Tự kiểm tra: không thấy khiếm khuyết nào."
    parts = [f"Tự kiểm tra: {report.found} khiếm khuyết, {report.calls} lời gọi, vá {report.patched} mục"]
    if report.blocked:
        parts.append(f"bỏ qua {report.blocked} mục model đã không sửa được nhiều lần (nên soát tay)")
    if report.remaining:
        parts.append(f"còn {report.remaining} mục chưa xử lý")
    return ", ".join(parts) + "."


_LABEL_IN_DETAIL = re.compile(r"^'(.+?)' xuất hiện")


def defect_units(defect: AnalysisDefect) -> list[str]:
    """Các "đơn vị" khiếm khuyết (khoá ổn định) — để nhớ cái nào model đã bỏ qua.

    Khiếm khuyết theo dòng tách thành MỘT đơn vị cho mỗi dòng (``self_address:123``) nên sửa
    được một phần danh sách vẫn nhận ra phần còn lại; ``unlisted_label`` theo nhãn; loại ở cấp
    bảng theo loại + mô tả.
    """
    if defect.kind == "unlisted_label":
        match = _LABEL_IN_DETAIL.match(defect.detail)
        return [f"unlisted_label:{match.group(1) if match else defect.detail}"]
    if defect.line_ids:
        return [f"{defect.kind}:{line_id}" for line_id in defect.line_ids]
    return [f"{defect.kind}:{defect.detail}"]


def _retarget(defect: AnalysisDefect, line_ids: tuple[int, ...]) -> AnalysisDefect:
    """Bản của ``defect`` chỉ gồm ``line_ids`` (mô tả khớp số dòng thật)."""
    if defect.kind == "missing_speaker":
        detail = f"{len(line_ids)} dòng có lời nhưng chưa có người nói."
    elif defect.kind == "self_address":
        detail = f"{len(line_ids)} dòng ghi người nghe trùng người nói (nghi sai)."
    else:
        detail = defect.detail
    return AnalysisDefect(defect.kind, detail, line_ids)


def filter_declined(
    defects: Sequence[AnalysisDefect], declined: Mapping[str, int], limit: int = DECLINE_LIMIT
) -> list[AnalysisDefect]:
    """Bỏ các đơn vị khiếm khuyết đã bị model bỏ qua từ ``limit`` lần trở lên.

    Khiếm khuyết theo dòng còn lại những dòng chưa bị bỏ qua; hết dòng thì bỏ cả khiếm khuyết.
    """
    kept: list[AnalysisDefect] = []
    for defect in defects:
        if defect.kind == STALE_GLOSSARY_KIND:
            kept.append(defect)
        elif defect.line_ids and defect.kind != "unlisted_label":
            ids = tuple(
                i for i in defect.line_ids if declined.get(f"{defect.kind}:{i}", 0) < limit
            )
            if ids:
                kept.append(defect if len(ids) == len(defect.line_ids) else _retarget(defect, ids))
        elif all(declined.get(unit, 0) < limit for unit in defect_units(defect)):
            kept.append(defect)
    return kept


def note_declined(
    declined: Mapping[str, int], sent: Sequence[AnalysisDefect],
    remaining: Sequence[AnalysisDefect],
) -> dict[str, int]:
    """Sổ "đã bỏ qua" mới: đơn vị đã gửi mà vẫn còn sau bản vá được cộng một lần.

    Gợi ý ``stale_glossary`` không bao giờ ghi (model không có việc phải làm với chúng).
    """
    still = {
        unit for d in remaining if d.kind != STALE_GLOSSARY_KIND for unit in defect_units(d)
    }
    updated = dict(declined)
    for defect in sent:
        if defect.kind == STALE_GLOSSARY_KIND:
            continue
        for unit in defect_units(defect):
            if unit in still:
                updated[unit] = updated.get(unit, 0) + 1
    return updated


def audit_film_id(source_lines: Sequence[Any], target_lang: str) -> str:
    """Dấu của phim + ngôn ngữ đích (khoá sổ "đã bỏ qua"); đổi phụ đề thì sổ mới."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update((target_lang or "").encode("utf-8"))
    for line in source_lines:
        digest.update(f"{getattr(line, 'index', '')}|{_line_text(line)}\n".encode())
    return digest.hexdigest()[:20]


def plan_audit_batches(
    defects: Sequence[AnalysisDefect],
    line_order: Sequence[int],
    window: int = AUDIT_WINDOW_LINES,
    max_lines: int = AUDIT_MAX_PROMPT_LINES,
) -> list[list[AnalysisDefect]]:
    """Chia khiếm khuyết thành các lô, mỗi lô có cửa sổ dòng nguồn ≤ ``max_lines``.

    Dòng nghi lỗi xếp theo thứ tự phim rồi gom dần vào lô chừng nào hợp các cửa sổ ``±window``
    còn trong trần. Khiếm khuyết cấp bảng (không có dòng: bảng rỗng, thuật ngữ cũ…) vào lô đầu.

    Args:
        defects: Khiếm khuyết cần gửi.
        line_order: Số dòng nguồn theo thứ tự phim (để tính vị trí/cửa sổ).
        window: Số dòng mỗi phía.
        max_lines: Trần số dòng nguồn mỗi lô.

    Returns:
        Danh sách lô (mỗi lô là danh sách :class:`AnalysisDefect`); rỗng nếu không có gì gửi.
    """
    position = {line_id: i for i, line_id in enumerate(line_order)}
    table_level = [d for d in defects if not d.line_ids]
    pairs = sorted(
        ((position[i], d, i) for d in defects for i in d.line_ids if i in position),
        key=lambda item: item[0],
    )
    groups: list[list[tuple[int, AnalysisDefect, int]]] = []
    covered: set[int] = set()
    current: list[tuple[int, AnalysisDefect, int]] = []
    for item in pairs:
        span = set(range(max(0, item[0] - window), min(len(line_order), item[0] + window + 1)))
        if current and len(covered | span) > max_lines:
            groups.append(current)
            current, covered = [], set()
        current.append(item)
        covered |= span
    if current:
        groups.append(current)
    batches: list[list[AnalysisDefect]] = []
    for group in groups:
        per_defect: dict[int, list[int]] = {}
        by_object: dict[int, AnalysisDefect] = {}
        for _pos, defect, line_id in group:
            per_defect.setdefault(id(defect), []).append(line_id)
            by_object[id(defect)] = defect
        batches.append([
            _retarget(by_object[key], tuple(ids)) for key, ids in per_defect.items()
        ])
    if table_level:
        if batches:
            batches[0] = [*table_level, *batches[0]]
        else:
            batches.append(list(table_level))
    return batches


def defects_to_prompt(defects: list[AnalysisDefect]) -> str:
    """Danh sách khiếm khuyết dạng JSON gọn cho prompt."""
    return json.dumps(
        [{"loai": d.kind, "chi_tiet": d.detail, "dong": list(d.line_ids)} for d in defects],
        ensure_ascii=False,
    )


__all__ = [
    "AUDIT_MAX_CALLS", "AUDIT_MAX_PROMPT_LINES", "AUDIT_THINKING_LEVEL", "AUDIT_WINDOW_LINES",
    "DECLINE_LIMIT", "DEFAULT_AUDIT_MODEL", "MAX_AUDIT_ROUNDS", "NO_ADDRESS_RULES_KIND",
    "STALE_GLOSSARY_KIND",
    "AnalysisDefect", "AuditReport", "apply_audit_patch", "describe_audit_report", "audit_film_id", "defect_units", "defects_to_prompt",
    "filter_declined", "find_analysis_defects", "note_declined", "plan_audit_batches",
]
