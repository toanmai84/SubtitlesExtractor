"""[v3.23.562] Quy nhãn người nói/người nghe của Visual Cues về TÊN trong bảng nhân vật.

Chuẩn đoán thật v3.23.561: quét bổ sung cues phủ lên 99,7% số dòng, nhưng bộ quét không
được đưa bảng nhân vật nên 95% nhãn là vai vế chung chung — nhiều nhãn chép nguyên ví
dụ trong prompt: "Nữ tỳ" 208 dòng, "Nam đệ tử" 131, "Sư phụ" 24, trong một phim đô thị
hiện đại. Nhãn không khớp bảng nhân vật thì mọi lưới xưng hô (tra quy tắc theo tên) đều
bỏ qua, và khâu dịch còn bị gợi ý chọn xưng hô tôi tớ/sư đồ.

Mô-đun này (hàm thuần) làm hai việc:

* Quy nhãn khớp tên gốc, bí danh hoặc phiên âm trong bảng nhân vật về TÊN CHUẨN
  (trường ``name`` của bảng), để lưới xưng hô và bảng quy đổi tên dùng được.
* Cho biết một nhãn có phải nhân vật trong bảng không — để đo độ phủ cues dùng được.
"""

from __future__ import annotations

import re
from dataclasses import replace

from subtitles_extractor.application.services.roster_format import (
    normalize_punctuation,
    parse_roster,
)
from subtitles_extractor.domain.ports.subtitle_translator_port import VisualCue
from subtitles_extractor.domain.value_objects.speaker_label import (
    canonical_fallback_speaker,
)

_ALIAS_SPLIT_RE = re.compile(r"\s*[/|]\s*")
#: Ký tự rác ở mép nhãn (gạch nối, gạch ngang, dấu hai chấm, chấm giữa, phẩy).
_LABEL_EDGE_CHARS = " -\u2013\u2014:\u00b7,"
#: [v3.23.596] Trợ từ sở hữu viết khác nhau của cùng một nhãn (之 Hán văn, の Nhật).
_POSSESSIVE_PARTICLES = str.maketrans({"之": "的", "の": "的"})
_CJK_CHAR = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_NAME_RE = re.compile(f"^[{_CJK_CHAR}]{{2,6}}$")
#: [v3.23.599] Chữ Hán liền chữ cái Latinh ("王Đại Phúc") → chèn khoảng trắng.
_CJK_LATIN_BOUNDARY_RE = re.compile(f"(?<=[{_CJK_CHAR}])(?=[A-Za-zÀ-ỹ])")


def _label_key(label: str) -> str:
    """Khoá so khớp: bỏ chú thích trong ngoặc, gộp khoảng trắng, không phân biệt hoa.

    [v3.23.596] Trợ từ sở hữu 之/の quy về 的: bộ quét ghi cùng một vai lúc "洪震的姘頭",
    lúc "洪震之姘頭", lúc "洪震の姘頭" (chữ Nhật) — lượt thật 10/10/2026 tách một nhân vật
    thành hai vai lồng tiếng với hai giọng khác nhau, và quy tắc xưng hô không tra được.
    """
    text = normalize_punctuation(label or "")
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.translate(_POSSESSIVE_PARTICLES)
    text = _CJK_LATIN_BOUNDARY_RE.sub(" ", text)
    return " ".join(text.split()).strip(_LABEL_EDGE_CHARS).casefold()


def _mixed_spellings(names: list[str], readings: list[str]) -> list[str]:
    """[v3.23.599] Nhãn LAI chữ Hán + phiên âm của một tên: "王 Đại Phúc", "王大 Phúc".

    Bộ quét bổ sung đôi khi giữ họ bằng chữ Hán và viết tên bằng phiên âm. Lượt dịch thật
    phim "he thong sung the" (``217d710a``, ``069d0c24``): "王 Đại Phúc" 12 dòng,
    "王 Mai" 3 dòng — không khớp bảng nhân vật (王大福 / Vương Đại Phúc, 王梅 /
    Vương Mai) nên thành VAI RIÊNG khi lồng tiếng (giọng khác chính nhân vật đó) và
    khâu dịch không tra được quy tắc xưng hô. Ghép theo SỐ ÂM TIẾT: bí danh n chữ Hán
    chỉ đi với phiên âm n âm tiết của CÙNG nhân vật, nên không thể ghép sang người khác.

    Args:
        names: Bí danh chữ Hán của một nhân vật.
        readings: Các phiên âm của nhân vật đó.

    Returns:
        Các cách viết lai (chữ Hán đầu, phiên âm phần còn lại).
    """
    spellings: list[str] = []
    for name in names:
        if not _CJK_NAME_RE.match(name):
            continue
        for reading in readings:
            syllables = reading.split()
            if len(syllables) != len(name):
                continue
            spellings += [
                f"{name[:cut]} {' '.join(syllables[cut:])}" for cut in range(1, len(name))
            ]
    return spellings


def roster_label_map(characters: str) -> dict[str, str]:
    """Ánh xạ mọi cách gọi một nhân vật (khoá chuẩn hoá) → tên chuẩn trong bảng.

    Cách gọi gồm: cả trường ``name`` (``"吴二狗 / 土狗"``), từng bí danh (``"土狗"``)
    và phần phiên âm (``"Tống Tiền"``); [v3.23.599] cả cách viết lai họ chữ Hán + tên
    phiên âm (``"王 Đại Phúc"``). Khi hai nhân vật tranh cùng một khoá, khoá đó bị bỏ để
    không gán nhầm người. Hàm thuần.

    Args:
        characters: Bảng nhân vật (văn bản).

    Returns:
        Dict ``khoá → tên chuẩn``; rỗng nếu không có bảng nhân vật.
    """
    claims: dict[str, set[str]] = {}
    for entry in parse_roster(characters or ""):
        canonical = entry.name.strip()
        if not canonical:
            continue
        names = [part.strip() for part in _ALIAS_SPLIT_RE.split(canonical)]
        readings = [part.strip() for part in _ALIAS_SPLIT_RE.split(entry.aka or "")]
        spellings = [canonical, *names, *readings, *_mixed_spellings(names, readings)]
        for spelling in spellings:
            key = _label_key(spelling)
            if key:
                claims.setdefault(key, set()).add(canonical)
    return {key: next(iter(names)) for key, names in claims.items() if len(names) == 1}


def canonical_label(label: str, label_map: dict[str, str]) -> str:
    """Tên chuẩn của ``label`` nếu khớp bảng nhân vật; ngược lại giữ nguyên nhãn.

    Args:
        label: Nhãn thô (``spk`` hoặc ``to``).
        label_map: Kết quả :func:`roster_label_map`.

    Returns:
        Tên chuẩn, hoặc chính ``label`` nếu không khớp.
    """
    # [v3.23.600] Nhãn dự phòng theo giọng viết lệch ("giọng nam", "Giọng nữ (chưa rõ)")
    # quy về một nhãn — không thì thành nhiều vai lồng tiếng cho cùng một ý.
    return (
        _lookup(label, label_map)
        or canonical_fallback_speaker(_label_key(label))
        or label
    )


def is_roster_label(label: str, label_map: dict[str, str]) -> bool:
    """Nhãn có trỏ tới một nhân vật trong bảng không."""
    return bool(label) and bool(_lookup(label, label_map))


def _lookup(label: str, label_map: dict[str, str]) -> str:
    """Tên chuẩn của nhãn; rỗng nếu không khớp.

    [v3.23.562] Nhãn ghi bí danh ĐẢO thứ tự ("赵可可 / 可可" trong khi bảng ghi
    "可可 / 赵可可" — 8 nhãn ở dữ liệu thật) vẫn khớp khi mọi bí danh cùng trỏ một người.
    """
    direct = label_map.get(_label_key(label))
    if direct:
        return direct
    parts = [part for part in _ALIAS_SPLIT_RE.split(label or "") if part.strip()]
    if len(parts) < 2:
        return ""
    names = {label_map.get(_label_key(part), "") for part in parts}
    return next(iter(names)) if len(names) == 1 and "" not in names else ""


def canonicalize_cue_labels(cues: list[VisualCue], characters: str) -> list[VisualCue]:
    """Quy nhãn ``speaker``/``addressee`` của mọi cue về tên chuẩn trong bảng nhân vật.

    Args:
        cues: Danh sách cue.
        characters: Bảng nhân vật.

    Returns:
        Danh sách cue mới (cue không đổi được giữ nguyên đối tượng).
    """
    label_map = roster_label_map(characters)
    if not label_map:
        return list(cues)
    result: list[VisualCue] = []
    for cue in cues:
        speaker = canonical_label(cue.speaker, label_map) if cue.speaker else ""
        addressee = canonical_label(cue.addressee, label_map) if cue.addressee else ""
        if speaker == cue.speaker and addressee == cue.addressee:
            result.append(cue)
        else:
            result.append(replace(cue, speaker=speaker, addressee=addressee))
    return result


def usable_cue_lines(cues: list[VisualCue], characters: str) -> set[int]:
    """Các dòng có cue DÙNG ĐƯỢC cho xưng hô.

    Có bảng nhân vật: người nói HOẶC người nghe phải là nhân vật trong bảng — nhãn vai vế
    chung chung ("Nữ phụ", "Nam thanh niên") không tra được quy tắc xưng hô nào. Không có
    bảng nhân vật: mọi cue có nhãn đều tính.

    Args:
        cues: Danh sách cue.
        characters: Bảng nhân vật.

    Returns:
        Tập chỉ số dòng.
    """
    label_map = roster_label_map(characters)
    if not label_map:
        return {cue.line_no for cue in cues if cue.speaker or cue.addressee}
    return {
        cue.line_no
        for cue in cues
        if is_roster_label(cue.speaker, label_map)
        or is_roster_label(cue.addressee, label_map)
    }


def roster_dialogue_cues_json(visual_cues: str, characters: str) -> str:
    """[v3.23.562] Chỉ giữ cue mà CẢ người nói lẫn người nghe là nhân vật trong bảng.

    Dùng cho số liệu xưng hô theo cặp: nhãn chung chung ("Nữ chính", "Nam chính") gom
    nhiều người khác nhau vào một "cặp", sinh cờ trôi xưng hô ảo. Ca thật v3.23.561: cặp
    "Nữ chính → Nam chính" bị báo dùng lẫn em/chị/ta/anh vì thực ra là nhiều cặp nhân vật.

    Args:
        visual_cues: Chuỗi JSON Visual Cues.
        characters: Bảng nhân vật; rỗng thì trả nguyên ``visual_cues``.

    Returns:
        Chuỗi JSON cues đã lọc và quy nhãn về tên chuẩn.
    """
    from subtitles_extractor.application.services.visual_cue_serializer import (
        deserialize_visual_cues,
        serialize_visual_cues,
    )

    label_map = roster_label_map(characters)
    if not label_map or not (visual_cues or "").strip():
        return visual_cues
    cues = canonicalize_cue_labels(deserialize_visual_cues(visual_cues), characters)
    kept = [
        cue
        for cue in cues
        if is_roster_label(cue.speaker, label_map)
        and is_roster_label(cue.addressee, label_map)
    ]
    return serialize_visual_cues(kept)
