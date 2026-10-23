"""[v3.23.593] Phân vai LỒNG TIẾNG: dòng nào ai nói, nhân vật nào đọc bằng giọng nào.

Trước bản này trang TTS đọc cả phim bằng MỘT giọng — bản thuyết minh, không phải lồng
tiếng. Dữ liệu để lồng tiếng đã có sẵn ở khâu phân tích ngữ cảnh: Visual Cues ghi người
nói từng dòng, bảng nhân vật ghi giới tính và tuổi. Lượt dịch thật v3.23.592 (phim
"灭门惨案II", 1246 dòng): cues phủ 1245 dòng, 99,4% nhãn người nói khớp bảng nhân vật;
18 nhân vật (13 nam, 5 nữ), ba vai chính nói 262/224/204 câu.

Mô-đun gồm các hàm THUẦN:

* :func:`line_speakers` — người nói của từng dòng TTS. Ưu tiên tag ``[Tên:]`` đầu dòng
(người
  dùng sửa tay ở bản dịch), sau đó Visual Cues: cue đánh số theo câu GỐC, nên dòng TTS
  được ghép với câu gốc trùng thời gian nhiều nhất (bản dịch có thể đã chỉnh mốc thời
  gian).
* :func:`build_cast` — danh sách vai kèm giới tính, nhóm tuổi, số câu.
* :func:`auto_cast` — giao giọng: đúng giới tính; vai nhiều câu chọn trước; khi phải dùng
  chung giọng thì chọn giọng của người ÍT nói chuyện trực tiếp với vai này nhất, để hai
  người đối đáp không cùng một giọng.
* :func:`voices_for_lines` — giọng từng dòng cho ``TTSRequest.line_voices``.
* [v3.23.597] Chế độ CHỈ HAI GIỌNG (nam/nữ): :func:`default_gender_voices`,
  :func:`two_voice_sides`, :func:`two_voice_casting`.
* [v3.23.598] :func:`role_choices` — vai để chọn khi người dùng sửa tay người nói một
  dòng (sửa tay nằm ở ``dubbing_line_edits``).
"""

from __future__ import annotations

import bisect
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import pairwise

from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonical_label,
    canonicalize_cue_labels,
    roster_label_map,
)
from subtitles_extractor.application.services.fill_support_guard import (
    demote_unsupported_scan_labels,
)
from subtitles_extractor.application.services.roster_format import (
    RosterEntry,
    parse_roster,
)
from subtitles_extractor.application.services.visual_cue_serializer import (
    deserialize_visual_cues,
)
from subtitles_extractor.application.services.vocative_guard import (
    fix_vocative_cue_labels,
)
from subtitles_extractor.domain.value_objects.speaker_label import (
    fallback_speaker_gender,
)
from subtitles_extractor.domain.value_objects.voice_profile import (
    TONE_HIGH,
    TONE_LOW,
    VOICE_FEMALE,
    VOICE_MALE,
    VoiceProfile,
    gender_from_roster,
    style_rank,
)

#: Nhóm tuổi của vai (quyết định giọng trầm/cao khi engine có biến thể cao độ).
AGE_CHILD = "child"
AGE_ADULT = "adult"
AGE_ELDER = "elder"
_CHILD_MAX_AGE = 14
#: [v3.23.594] Dòng TTS "khớp" câu gốc khi mốc bắt đầu lệch không quá ngưỡng này (sau khi
#: trừ độ lệch chung), và phải có ít nhất ``ALIGN_MIN_RATIO`` dòng khớp mới coi là cùng
#: phim. Đo trên 3 phim của kho: cùng phim (kể cả dời giờ cả bộ ±3 giây, xoá/tách 5% dòng)
#: khớp 96-100%; ghép chéo phim khác khớp 1-57%.
ALIGN_TOLERANCE_S = 0.5
ALIGN_MIN_RATIO = 0.8
_ELDER_MIN_AGE = 60
#: [v3.23.597] Cách phân giọng: mỗi nhân vật một giọng, hoặc chỉ hai giọng (nam/nữ).
CAST_MODE_PER_ROLE = "per_role"
CAST_MODE_TWO_VOICES = "two_voices"
CAST_MODES = (CAST_MODE_PER_ROLE, CAST_MODE_TWO_VOICES)
#: Số câu gốc liền trước được xét khi tìm câu trùng thời gian (câu gốc dài có thể bắt đầu
#: sớm hơn vài câu ngắn).
_OVERLAP_LOOKBACK = 16

_SPEAKER_TAG_RE = re.compile(r"^\s*\[([^\[\]]+?)\s*[:：]\]")  # noqa: RUF001 — dấu CJK cố ý
_AGE_NUMBER_RE = re.compile(r"(\d{1,3})")
_ELDER_WORDS = ("già", "lão", "cụ", "cao tuổi", "elder", "old")
_CHILD_WORDS = (
    "cậu bé", "cô bé", "em bé", "bé gái", "bé trai", "trẻ con", "trẻ em", "nhi đồng",
)
_FEMALE_LEAD_WORDS = frozenset({
    "bà", "cô", "chị", "mẹ", "nữ", "vợ", "thím", "dì", "mợ", "nàng",
})
_MALE_LEAD_WORDS = frozenset({
    "ông", "anh", "chú", "bác", "lão", "cậu", "bố", "cha", "chồng", "nam", "gã",
})
#: [v3.23.596] Gợi ý giới tính trong nhãn CHỮ HÁN của vai ngoài bảng nhân vật. Bộ quét
#: Visual Cues hay ghi vai phụ bằng chữ gốc ("老伯", "大嫂"); trước đây chỉ đọc chữ đầu
#: tiếng Việt nên mọi nhãn chữ Hán thành "không rõ giới tính" và có thể nhận giọng khác
#: giới — lượt lồng tiếng thật 10/10/2026: "老伯" (ông bác già, mô tả cue "Ông cụ…") được
#: giao giọng nữ. Cụm nhiều chữ đứng trước để "女婿" (con rể) không bị chữ "女" bắt nhầm.
_CJK_GENDER_HINTS: tuple[tuple[str, str], ...] = (
    ("女婿", VOICE_MALE), ("老公", VOICE_MALE), ("公公", VOICE_MALE),
    ("先生", VOICE_MALE), ("丈夫", VOICE_MALE), ("和尚", VOICE_MALE),
    ("姐夫", VOICE_MALE), ("妹夫", VOICE_MALE), ("姨夫", VOICE_MALE),
    ("姑夫", VOICE_MALE), ("夫君", VOICE_MALE),
    ("公主", VOICE_FEMALE), ("老婆", VOICE_FEMALE), ("夫人", VOICE_FEMALE),
    ("太太", VOICE_FEMALE), ("小姐", VOICE_FEMALE), ("姑娘", VOICE_FEMALE),
    *((char, VOICE_MALE) for char in "男爷爺伯叔哥兄弟爸父舅郎僧翁"),
    *((char, VOICE_FEMALE) for char in "女妈媽母娘婆嫂姐妹姑姨婶嬸奶妻媳妃尼妇婦"),
)
#: [v3.23.614] Từ có chữ chỉ giới nhưng KHÔNG chỉ giới: "弟子" (đệ tử), "徒弟" (đồ đệ),
#: "子弟" (con em), "大夫" (thầy thuốc). Ca thật 14/10: "女弟子" đọc giọng nam vì
#: chữ "弟".
_CJK_NEUTRAL_WORDS = ("弟子", "徒弟", "子弟", "大夫")
_CJK_ELDER_HINTS = (
    "老伯", "老头", "老頭", "老太", "老爷", "老爺", "老翁", "老汉", "老漢", "老妇",
    "老婦", "爷爷", "爺爺", "奶奶", "婆婆", "阿婆", "阿公",
)
_CJK_CHILD_HINTS = ("小孩", "孩子", "男孩", "女孩", "小童", "儿童", "兒童", "娃娃")


@dataclass(frozen=True, slots=True)
class DubbingSource:
    """Ngữ cảnh phân vai từ trang Dịch.

    Attributes:
        characters: Bảng nhân vật (văn bản).
        visual_cues: JSON Visual Cues (``id`` = vị trí câu GỐC, đánh số từ 1).
        source_spans: ``(start_sec, end_sec)`` của từng câu gốc, theo thứ tự.
        source_texts: [v3.23.599] Chữ từng câu gốc, theo thứ tự — để soát người nói
            (câu gốc gọi tên chính người được gán là người nói).
    """

    characters: str = ""
    visual_cues: str = ""
    source_spans: tuple[tuple[float, float], ...] = ()
    source_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CastMember:
    """Một vai trong phim.

    Attributes:
        name: Nhãn chuẩn (tên trong bảng nhân vật, hoặc nhãn người nói của cue).
        display_name: Tên đọc được cho người dùng (phiên âm tiếng Việt nếu có).
        gender: :data:`VOICE_MALE`, :data:`VOICE_FEMALE` hoặc rỗng.
        age_group: :data:`AGE_CHILD`, :data:`AGE_ADULT`, :data:`AGE_ELDER` hoặc rỗng.
        line_count: Số dòng vai này nói.
        in_roster: Có trong bảng nhân vật không.
    """

    name: str
    display_name: str
    gender: str
    age_group: str
    line_count: int
    in_roster: bool


def line_speakers(events: list[object], source: DubbingSource) -> list[str]:
    """Người nói của từng dòng TTS (rỗng = không biết, đọc bằng giọng chung).

    Args:
        events: Dòng TTS (có ``text``, ``start_sec``, ``end_sec``).
        source: Ngữ cảnh phân vai.

    Returns:
        Danh sách nhãn chuẩn, cùng độ dài ``events``.
    """
    label_map = roster_label_map(source.characters)
    cue_speakers = _speakers_by_source_position(source)
    positions = align_to_source(events, source.source_spans)
    speakers: list[str] = []
    for event, source_position in zip(events, positions, strict=True):
        tag = _SPEAKER_TAG_RE.match(str(getattr(event, "text", "") or ""))
        if tag is not None:
            speakers.append(canonical_label(tag.group(1).strip(), label_map))
        elif source_position is None:
            speakers.append("")
        else:
            speakers.append(cue_speakers.get(source_position, ""))
    return speakers


def align_to_source(
    events: list[object],
    spans: tuple[tuple[float, float], ...] | list[tuple[float, float]],
) -> list[int | None]:
    """Vị trí câu GỐC (trang Dịch) của từng dòng TTS; toàn ``None`` nếu không cùng phim.

    [v3.23.594] Visual Cues chỉ đúng với phụ đề của CHÍNH phim đã phân tích. Bản 593 ghép
    mọi dòng TTS vào câu gốc trùng thời gian nhiều nhất mà không hỏi hai bên có cùng phim
    không: nạp phụ đề phim khác, hay "Tổng hợp cả bộ" các tập khác, thì dòng nào cũng
    "trùng" một câu gốc nào đó — thử phụ đề phim "he thong sung the" (5540 dòng) với bảng
    vai phim "灭门惨案II": 1488 dòng nhận giọng vai của phim kia.

    Nay chỉ ghép khi ≥ :data:`ALIGN_MIN_RATIO` dòng TTS bắt đầu cách một câu gốc không quá
    :data:`ALIGN_TOLERANCE_S` (sau khi trừ độ lệch chung — phụ đề bị dời giờ cả bộ vẫn
    nhận). Cùng số dòng và khớp mốc thì ghép THEO VỊ TRÍ; khác số dòng (đã xoá/tách dòng)
    thì ghép theo thời gian trùng nhiều nhất.

    Args:
        events: Dòng TTS (có ``start_sec``, ``end_sec``).
        spans: ``(start, end)`` từng câu gốc theo thứ tự.

    Returns:
        Vị trí câu gốc (0-based) hoặc ``None`` cho từng dòng.
    """
    starts = [float(getattr(event, "start_sec", 0.0)) for event in events]
    ends = [float(getattr(event, "end_sec", 0.0)) for event in events]
    unmatched: list[int | None] = [None] * len(events)
    if not events or not spans:
        return unmatched
    if len(events) == len(spans):
        offsets = [start - span[0] for start, span in zip(starts, spans, strict=True)]
        if _matched_ratio(offsets) >= ALIGN_MIN_RATIO:
            return list(range(len(events)))
    order = sorted(range(len(spans)), key=lambda pos: spans[pos][0])
    source_starts = [spans[pos][0] for pos in order]
    offsets = [_nearest_offset(start, source_starts) for start in starts]
    if _matched_ratio(offsets) < ALIGN_MIN_RATIO:
        return unmatched
    shift = statistics.median(offsets)
    span_list = list(spans)
    return [
        _overlapping_source_position(
            start - shift, end - shift, span_list, order, source_starts
        )
        for start, end in zip(starts, ends, strict=True)
    ]


def _matched_ratio(offsets: list[float]) -> float:
    """Tỉ lệ dòng lệch khỏi độ lệch CHUNG (trung vị) không quá ngưỡng."""
    if not offsets:
        return 0.0
    shift = statistics.median(offsets)
    matched = sum(1 for offset in offsets if abs(offset - shift) <= ALIGN_TOLERANCE_S)
    return matched / len(offsets)


def _nearest_offset(start: float, source_starts: list[float]) -> float:
    """Độ lệch có dấu tới mốc bắt đầu câu gốc gần nhất."""
    index = bisect.bisect_left(source_starts, start)
    candidates = [
        source_starts[i] for i in (index - 1, index) if 0 <= i < len(source_starts)
    ]
    return min((start - candidate for candidate in candidates), key=abs)


def build_cast(speakers: list[str], characters: str) -> list[CastMember]:
    """Danh sách vai có thoại, nhiều câu trước.

    Args:
        speakers: Kết quả :func:`line_speakers`.
        characters: Bảng nhân vật.

    Returns:
        Vai sắp theo số câu giảm dần (cùng số câu thì theo thứ tự xuất hiện).
    """
    counts = Counter(speaker for speaker in speakers if speaker)
    first_seen: dict[str, int] = {}
    for index, speaker in enumerate(speakers):
        first_seen.setdefault(speaker, index)
    roster = {entry.name.strip(): entry for entry in parse_roster(characters or "")}
    members = [
        _member(name, count, roster.get(name))
        for name, count in counts.items()
    ]
    members.sort(key=lambda member: (-member.line_count, first_seen[member.name]))
    return members


def conversation_weights(speakers: list[str]) -> dict[frozenset[str], int]:
    """Số lần hai vai nói NỐI TIẾP nhau (đo mức đối đáp trực tiếp).

    Args:
        speakers: Người nói theo thứ tự dòng.

    Returns:
        ``{frozenset({A, B}): số lần}``.
    """
    weights: Counter[frozenset[str]] = Counter()
    for current, following in pairwise(speakers):
        if current and following and current != following:
            weights[frozenset((current, following))] += 1
    return dict(weights)


def auto_cast(
    cast: list[CastMember],
    catalog: list[VoiceProfile],
    weights: dict[frozenset[str], int],
) -> dict[str, str]:
    """Giao giọng cho từng vai.

    Thứ tự ưu tiên khi chọn giọng cho một vai (vai nhiều câu chọn trước):

    1. Đúng giới tính (nếu biết và engine có giọng giới đó).
    2. Ít "đụng" nhất: tổng số lần vai này đối đáp với các vai ĐÃ giữ giọng đó.
    3. Ít người dùng nhất — còn giọng trống thì dùng giọng trống.
    4. Đúng sắc giọng theo tuổi (già → trầm, trẻ con → cao, còn lại → thường).

    Args:
        cast: Kết quả :func:`build_cast`.
        catalog: Giọng của engine.
        weights: Kết quả :func:`conversation_weights`.

    Returns:
        ``{tên vai: voice_id}``; rỗng nếu engine không có giọng nào.
    """
    if not catalog:
        return {}
    holders: dict[str, list[str]] = {profile.voice_id: [] for profile in catalog}
    casting: dict[str, str] = {}
    for member in cast:
        candidates = [
            profile for profile in catalog if profile.gender == member.gender
        ] if member.gender else []
        pool = candidates or catalog
        chosen = min(pool, key=partial(
            _voice_cost, name=member.name, holders=holders, weights=weights,
            preferred_tone=_preferred_tone(member.age_group),
        ))
        holders[chosen.voice_id].append(member.name)
        casting[member.name] = chosen.voice_id
    return casting


def _voice_cost(
    profile: VoiceProfile,
    *,
    name: str,
    preferred_tone: str,
    holders: dict[str, list[str]],
    weights: dict[frozenset[str], int],
) -> tuple[int, int, int, int, int]:
    """Khoá xếp hạng một giọng cho vai ``name`` (nhỏ hơn = hợp hơn).

    Returns:
        ``(số lần đối đáp với người đang giữ giọng, số người đang giữ, độ lệch phong cách,
        lệch sắc giọng, không phải giọng tuyển chọn)``.
    """
    others = holders[profile.voice_id]
    clash = sum(weights.get(frozenset((name, other)), 0) for other in others)
    # [v3.23.626] Giữa các giọng ngang nhau về đụng độ/số người dùng: ưu tiên giọng "tự nhiên"
    # (thoại nhân vật) hơn giọng đọc tin tức, rồi sắc giọng theo tuổi, rồi giọng tuyển chọn.
    return (
        clash, len(others), style_rank(profile),
        int(profile.tone != preferred_tone), int(not profile.featured),
    )


def voices_for_lines(speakers: list[str], casting: dict[str, str]) -> tuple[str, ...]:
    """Giọng từng dòng cho ``TTSRequest.line_voices``.

    Args:
        speakers: Người nói theo thứ tự dòng TTS.
        casting: ``{tên vai: voice_id}``.

    Returns:
        Tuple cùng độ dài ``speakers``; rỗng khi không dòng nào có giọng riêng.
    """
    voices = tuple(casting.get(speaker, "") if speaker else "" for speaker in speakers)
    return voices if any(voices) else ()


def role_choices(cast: list[CastMember], characters: str) -> list[CastMember]:
    """[v3.23.598] Vai để chọn khi sửa tay người nói một dòng.

    Vai đang có thoại (đúng thứ tự bảng phân vai), rồi nhân vật trong bảng nhân vật chưa
    được gán câu nào — dòng bị gán nhầm thường thuộc về một nhân vật ít thoại mà Visual
    Cues bỏ sót.

    Args:
        cast: Vai đang có (:func:`build_cast`).
        characters: Bảng nhân vật.

    Returns:
        Danh sách vai; nhân vật chưa có câu mang ``line_count == 0``.
    """
    choices = list(cast)
    named = {member.name for member in cast}
    for entry in parse_roster(characters or ""):
        name = entry.name.strip()
        if name and name not in named:
            named.add(name)
            choices.append(_member(name, 0, entry))
    return choices


def canonical_voice_id(voice: str, catalog: list[VoiceProfile]) -> str:
    """[v3.23.626] ID giọng hiện hành của ``voice`` (đã lưu dưới tên cũ thì đổi sang tên mới).

    Engine đổi tên giọng nhưng giữ tên cũ làm bí danh (``VoiceProfile.aliases``); lựa chọn đã
    lưu dưới tên cũ phải còn hiệu lực. ``voice`` đã là ID hợp lệ, hoặc không khớp bí danh nào
    thì giữ nguyên.
    """
    if any(profile.voice_id == voice for profile in catalog):
        return voice
    for profile in catalog:
        if voice in profile.aliases:
            return profile.voice_id
    return voice


def default_gender_voices(
    catalog: list[VoiceProfile], main_voice: str = ""
) -> tuple[str, str]:
    """[v3.23.597] Giọng nam và giọng nữ mặc định cho chế độ hai giọng.

    Giọng chung đang chọn ở tab Engine được giữ cho giới của nó (người dùng đã chọn giọng
    đó để đọc cả phim); giới còn lại lấy giọng đầu tiên của giới đó trong danh sách engine
    (VieNeu xếp giọng "biên tập viên chọn" lên đầu).

    Args:
        catalog: Giọng của engine.
        main_voice: Giọng chung đang chọn (rỗng = không có).

    Returns:
        ``(giọng nam, giọng nữ)``; rỗng nếu engine không có giọng nào. Engine không ghi
        giới tính thì lấy hai giọng đầu danh sách.
    """
    if not catalog:
        return "", ""
    main = next((p for p in catalog if p.voice_id == main_voice), None)

    def _pick(gender: str, fallback_index: int) -> str:
        if main is not None and main.gender == gender:
            return main.voice_id
        same = [profile for profile in catalog if profile.gender == gender]
        if same:
            # [v3.23.626] Giọng "tự nhiên" trước (thứ tự engine giữ khi ngang nhau): giọng
            # tuyển chọn đầu bảng của VieNeu có cả giọng đọc tin tức.
            return min(same, key=style_rank).voice_id
        return catalog[min(fallback_index, len(catalog) - 1)].voice_id

    return _pick(VOICE_MALE, 0), _pick(VOICE_FEMALE, 1)


def two_voice_sides(
    cast: list[CastMember],
    unknown_side: str = VOICE_MALE,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """[v3.23.597] Mỗi vai đọc bằng giọng nam hay giọng nữ.

    Thứ tự: lựa chọn của người dùng cho vai đó → giới tính của vai (bảng nhân vật / nhãn)
    → ``unknown_side`` cho vai chưa rõ giới tính.

    Args:
        cast: Vai trong phim.
        unknown_side: :data:`VOICE_MALE` hoặc :data:`VOICE_FEMALE` cho vai chưa rõ giới.
        overrides: ``{tên vai: giới}`` người dùng tự chọn.

    Returns:
        ``{tên vai: VOICE_MALE | VOICE_FEMALE}``.
    """
    chosen = overrides or {}
    sexes = (VOICE_MALE, VOICE_FEMALE)
    fallback = unknown_side if unknown_side in sexes else VOICE_MALE
    sides: dict[str, str] = {}
    for member in cast:
        side = chosen.get(member.name, "")
        if side not in sexes:
            side = member.gender or fallback
        sides[member.name] = side
    return sides


def two_voice_casting(
    sides: dict[str, str], male_voice: str, female_voice: str
) -> dict[str, str]:
    """[v3.23.597] ``{tên vai: voice_id}`` khi cả phim chỉ dùng hai giọng.

    Args:
        sides: Kết quả :func:`two_voice_sides`.
        male_voice: Giọng nam.
        female_voice: Giọng nữ.

    Returns:
        Bảng phân vai (dùng chung :func:`voices_for_lines`).
    """
    return {
        name: female_voice if side == VOICE_FEMALE else male_voice
        for name, side in sides.items()
    }


def _speakers_by_source_position(source: DubbingSource) -> dict[int, str]:
    """``{vị trí câu gốc (0-based): nhãn chuẩn}`` từ Visual Cues.

    [v3.23.602] Nhãn quét/hỏi lại đã lưu mà người nói không có mặt trong cảnh được hạ
    xuống nhãn giọng (cùng lưới với khâu dịch) — vai bịa không nhận giọng riêng.

    [v3.23.603] Nhãn đã lưu còn qua lưới dòng gọi tên của khâu dịch: phiên đã lưu ghi
    dòng đáp tên "秦可秀" (sau câu hỏi "你叫甚麼名字") là lời người hỏi — lồng tiếng đọc
    tên cô bằng giọng nam ở mọi lượt; nay sửa ngay khi nạp vai, không cần dịch lại.
    """
    cues = canonicalize_cue_labels(
        deserialize_visual_cues(source.visual_cues), source.characters
    )
    if source.source_texts:
        sources = {
            position + 1: text for position, text in enumerate(source.source_texts)
        }
        cues = fix_vocative_cue_labels(cues, sources, source.characters).cues
    cues, _demoted = demote_unsupported_scan_labels(cues, source.characters)
    return {cue.line_no - 1: cue.speaker.strip() for cue in cues if cue.speaker.strip()}


def _overlapping_source_position(
    start: float,
    end: float,
    spans: list[tuple[float, float]],
    order: list[int],
    starts: list[float],
) -> int | None:
    """Vị trí câu gốc trùng thời gian NHIỀU nhất với ``[start, end]``."""
    upper = bisect.bisect_left(starts, end)
    best_position: int | None = None
    best_overlap = 0.0
    for rank in range(upper - 1, max(-1, upper - 1 - _OVERLAP_LOOKBACK), -1):
        position = order[rank]
        span_start, span_end = spans[position]
        overlap = min(end, span_end) - max(start, span_start)
        if overlap > best_overlap:
            best_position, best_overlap = position, overlap
    return best_position


def _member(name: str, count: int, entry: RosterEntry | None) -> CastMember:
    """Dựng vai từ bảng nhân vật (nếu có) hoặc chỉ từ nhãn."""
    if entry is None:
        return CastMember(
            name=name, display_name=name, gender=_gender_from_label(name),
            age_group=_age_group("", name), line_count=count, in_roster=False,
        )
    readings = [
        part.strip() for part in re.split(r"\s*/\s*", entry.aka or "") if part.strip()
    ]
    return CastMember(
        name=name,
        display_name=readings[0] if readings else name,
        gender=gender_from_roster(entry.gender) or _gender_from_label(name),
        age_group=_age_group(entry.age, name),
        line_count=count,
        in_roster=True,
    )


def _gender_from_label(label: str) -> str:
    """Giới tính đoán từ nhãn vai ngoài bảng.

    Nhãn tiếng Việt: CHỮ ĐẦU ("Bà lão", "Ông chủ quán"). Nhãn chữ Hán: cụm/chữ chỉ giới
    (``老伯`` → nam, ``大嫂`` → nữ) — xem :data:`_CJK_GENDER_HINTS`.
    """
    # [v3.23.600] Nhãn dự phòng "Giọng nam chưa rõ"/"Giọng nữ chưa rõ" ghi đúng giới.
    fallback = fallback_speaker_gender(label)
    if fallback:
        return fallback
    words = (label or "").strip().casefold().split()
    if not words:
        return ""
    if words[0] in _FEMALE_LEAD_WORDS:
        return VOICE_FEMALE
    if words[0] in _MALE_LEAD_WORDS:
        return VOICE_MALE
    return _cjk_head_gender(label)


def _blank_neutral_words(text: str) -> str:
    """Xoá (thay bằng khoảng trắng) các từ không chỉ giới, TRỪ khi chúng chồng lên một cụm
    chỉ giới nhiều chữ (hàm thuần).

    [v3.23.615] Bản 3.23.614 xoá "大夫" (thầy thuốc) ở mọi nơi, nên "大夫人" (phu nhân cả)
    mất luôn cụm "夫人" và thành chưa rõ giới (bản 3.23.613 vẫn đọc đúng là nữ).

    Args:
        text: Nhãn vai.

    Returns:
        Nhãn cùng độ dài, các từ trung tính đứng riêng đã thành khoảng trắng.
    """
    compound_spans = [
        (start, start + len(hint))
        for hint, _gender in _CJK_GENDER_HINTS
        if len(hint) > 1
        for start in _find_all(text, hint)
    ]
    for word in _CJK_NEUTRAL_WORDS:
        for start in _find_all(text, word):
            end = start + len(word)
            if not any(left < end and start < right for left, right in compound_spans):
                text = text[:start] + " " * len(word) + text[end:]
    return text


def _find_all(text: str, word: str) -> list[int]:
    """Mọi vị trí bắt đầu của ``word`` trong ``text`` (kể cả chồng nhau)."""
    positions: list[int] = []
    start = text.find(word)
    while start != -1:
        positions.append(start)
        start = text.find(word, start + 1)
    return positions


def _cjk_head_gender(label: str) -> str:
    """Giới tính theo gợi ý chữ Hán đứng CUỐI CÙNG trong nhãn (hàm thuần).

    Danh từ chính của cụm chữ Hán đứng cuối: "伯母" là bác gái, "舅妈" là mợ, "姐夫"
    là anh rể — bản cũ lấy gợi ý đầu tiên theo thứ tự bảng (chữ nam trước) nên đọc
    ngược cả ba.
    Cụm nhiều chữ được ưu tiên hơn chữ đơn nằm trong nó ("女婿" là con rể).
    """
    text = _blank_neutral_words(label or "")
    best_end = -1
    best_gender = ""
    taken: list[tuple[int, int]] = []
    for hint, gender in _CJK_GENDER_HINTS:
        start = text.find(hint)
        while start != -1:
            end = start + len(hint)
            inside = any(left <= start and end <= right for left, right in taken)
            if not inside:
                if len(hint) > 1:
                    taken.append((start, end))
                if end > best_end:
                    best_end, best_gender = end, gender
            start = text.find(hint, start + 1)
    return best_gender


def _age_group(age: str, label: str) -> str:
    """Nhóm tuổi từ cột tuổi của bảng nhân vật ("~70", "trung niên") hoặc từ nhãn vai."""
    text = (age or "").casefold()
    number = _AGE_NUMBER_RE.search(text)
    if number is not None:
        years = int(number.group(1))
        if years <= _CHILD_MAX_AGE:
            return AGE_CHILD
        return AGE_ELDER if years >= _ELDER_MIN_AGE else AGE_ADULT
    if any(word in text for word in _ELDER_WORDS):
        return AGE_ELDER
    if "trung niên" in text:
        return AGE_ADULT
    folded_label = (label or "").casefold()
    if any(word in folded_label for word in _CHILD_WORDS + _CJK_CHILD_HINTS):
        return AGE_CHILD
    if folded_label.split()[:1] in (["lão"], ["cụ"]) or "ông lão" in folded_label:
        return AGE_ELDER
    if any(hint in folded_label for hint in _CJK_ELDER_HINTS):
        return AGE_ELDER
    return ""


def _preferred_tone(age_group: str) -> str:
    """Sắc giọng hợp nhóm tuổi."""
    if age_group == AGE_ELDER:
        return TONE_LOW
    if age_group == AGE_CHILD:
        return TONE_HIGH
    return ""


__all__ = [
    "AGE_ADULT",
    "AGE_CHILD",
    "AGE_ELDER",
    "ALIGN_MIN_RATIO",
    "ALIGN_TOLERANCE_S",
    "CAST_MODES",
    "CAST_MODE_PER_ROLE",
    "CAST_MODE_TWO_VOICES",
    "CastMember",
    "DubbingSource",
    "align_to_source",
    "auto_cast",
    "build_cast",
    "conversation_weights",
    "default_gender_voices",
    "line_speakers",
    "role_choices",
    "two_voice_casting",
    "two_voice_sides",
    "voices_for_lines",
]
