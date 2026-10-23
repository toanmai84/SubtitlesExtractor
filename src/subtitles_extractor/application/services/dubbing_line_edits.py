"""[v3.23.598] Sửa tay TỪNG DÒNG khi lồng tiếng: người nói (phân vai sai) và lời thoại.

Người nói từng dòng lấy từ nhãn Visual Cues của khâu phân tích ngữ cảnh, mà nhãn đó hay
lệch: trên 18 lượt phân tích thật phim "灭门惨案II", lượt mới nhất (10/10/2026) khác nhãn
đa số của các lượt ở 294/1241 dòng (23,7%); 80 dòng trong số đó khác cả GIỚI TÍNH — tức
đọc nhầm giọng nam/nữ. Ví dụ dòng 495-496: "阿秀，你幫我" (gọi A Tú nhờ giúp —
người nói là chồng cô) mang nhãn 秦可秀, câu đáp "好，我幫你" mang nhãn 甄劍聲,
nên bản dịch cũng sai theo nhãn ("Được, anh giúp em.").

Trước bản này chỉ sửa được bằng cách gõ tag ``[Tên:]`` vào bản dịch ở trang Dịch rồi nạp
lại; lời thoại chỉ sửa được SAU khi đã TTS cả phim; mọi chỉnh sửa mất khi nạp lại phụ đề.

Mô-đun gồm các hàm THUẦN:

* :func:`apply_speaker_overrides` — người nói sau khi áp lựa chọn tay.
* :func:`describe_line_speakers` — mỗi dòng: vai đang dùng, vai tự tìm, giọng, sửa tay?
* :func:`subtitle_fingerprint` — nhận ra "cùng một phụ đề" qua các lần nạp.
* :func:`restorable_text_edits` — chỉ khôi phục lời đã sửa khi dòng còn đúng chữ cũ.
* :func:`encode_line_edits` / :func:`decode_line_edits` / :func:`remember_line_edits` —
  lưu và đọc lại chỉnh sửa (JSON), giới hạn số phụ đề nhớ.
"""  # noqa: RUF002 — dấu phẩy chữ Hán trích nguyên văn phụ đề thật

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

#: Lựa chọn tay "giọng chung": dòng không thuộc vai nào, đọc bằng giọng ở tab Engine.
SHARED_VOICE_SPEAKER = ""
#: Số phụ đề nhớ chỉnh sửa tay (nhiều tập/phim; cũ nhất bị bỏ trước).
MAX_REMEMBERED_SUBTITLES = 40

_KEY_SPEAKERS = "speakers"
_KEY_TEXTS = "texts"
_KEY_LINES = "lines"


@dataclass(frozen=True, slots=True)
class TextEdit:
    """Lời thoại một dòng đã sửa tay.

    Attributes:
        original: Chữ của dòng lúc nạp (trước lần sửa đầu tiên).
        edited: Chữ người dùng sửa.
    """

    original: str
    edited: str


@dataclass(frozen=True, slots=True)
class LineSnapshot:
    """[v3.23.625] Một dòng của danh sách dòng SAU khi thêm/xoá dòng.

    Attributes:
        origin: Vị trí của dòng trong phụ đề NẠP GỐC; ``-1`` = dòng người dùng thêm.
        start_sec: Mốc bắt đầu.
        end_sec: Mốc kết thúc.
        text: Lời thoại hiện tại.
    """

    origin: int
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True, slots=True)
class LineEdits:
    """Mọi chỉnh sửa tay trên một phụ đề, theo VỊ TRÍ dòng (0-based).

    Attributes:
        speakers: ``{vị trí: tên vai}``; :data:`SHARED_VOICE_SPEAKER` = giọng chung.
        texts: ``{vị trí: TextEdit}``.
        lines: [v3.23.625] Danh sách dòng sau khi thêm/xoá dòng (``None`` = giữ nguyên các
            dòng lúc nạp). Khi có, ``speakers``/``texts`` đánh theo vị trí TRONG danh sách này.
    """

    speakers: dict[int, str] = field(default_factory=dict)
    texts: dict[int, TextEdit] = field(default_factory=dict)
    lines: tuple[LineSnapshot, ...] | None = None

    @property
    def is_empty(self) -> bool:
        """Không có chỉnh sửa nào."""
        return not self.speakers and not self.texts and self.lines is None


@dataclass(frozen=True, slots=True)
class LineSpeaker:
    """Người nói và giọng của một dòng.

    Attributes:
        speaker: Vai đang dùng (rỗng = giọng chung).
        auto_speaker: Vai ứng dụng tự tìm (tag ``[Tên:]`` hoặc Visual Cues).
        voice: Giọng của vai theo bảng phân vai (rỗng = giọng chung).
        overridden: Người dùng đã chọn tay người nói dòng này.
    """

    speaker: str
    auto_speaker: str
    voice: str
    overridden: bool


def apply_speaker_overrides(
    speakers: Sequence[str], overrides: Mapping[int, str]
) -> list[str]:
    """Người nói từng dòng sau khi áp lựa chọn tay.

    Args:
        speakers: Người nói tự tìm, theo vị trí dòng.
        overrides: ``{vị trí: tên vai}`` người dùng chọn; vị trí ngoài phạm vi bị bỏ qua.

    Returns:
        Danh sách mới cùng độ dài ``speakers``.
    """
    effective = list(speakers)
    for position, speaker in overrides.items():
        if 0 <= position < len(effective):
            effective[position] = speaker
    return effective


def describe_line_speakers(
    auto_speakers: Sequence[str],
    overrides: Mapping[int, str],
    casting: Mapping[str, str],
) -> list[LineSpeaker]:
    """Người nói, vai tự tìm và giọng của từng dòng (để hiện và sửa trên bảng dòng).

    Args:
        auto_speakers: Người nói tự tìm theo vị trí dòng.
        overrides: Lựa chọn tay ``{vị trí: tên vai}``.
        casting: Bảng phân vai ``{tên vai: voice_id}``.

    Returns:
        Một :class:`LineSpeaker` cho mỗi dòng.
    """
    effective = apply_speaker_overrides(auto_speakers, overrides)
    return [
        LineSpeaker(
            speaker=speaker,
            auto_speaker=auto,
            voice=casting.get(speaker, "") if speaker else "",
            overridden=position in overrides,
        )
        for position, (auto, speaker) in enumerate(
            zip(auto_speakers, effective, strict=True)
        )
    ]


def subtitle_fingerprint(events: Sequence[object]) -> str:
    """Dấu nhận ra CÙNG MỘT phụ đề qua các lần nạp: số dòng + mốc BẮT ĐẦU từng dòng.

    Không dùng chữ (bản dịch lại / lời sửa tay vẫn là phụ đề đó) và không dùng mốc kết
    thúc: trên ba lượt dịch thật phim "灭门惨案II", khâu dịch không đổi mốc bắt đầu
    dòng nào (0/1245) nhưng đổi mốc kết thúc 144-156 dòng, mỗi lượt một khác.

    Args:
        events: Dòng phụ đề (có ``start_sec``).

    Returns:
        Chuỗi hex 20 ký tự; phụ đề rỗng cho chuỗi rỗng.
    """
    if not events:
        return ""
    starts = ",".join(
        str(round(float(getattr(event, "start_sec", 0.0)) * 1000)) for event in events
    )
    digest = hashlib.sha1(f"{len(events)}|{starts}".encode(), usedforsecurity=False)
    return digest.hexdigest()[:20]


def restorable_text_edits(
    events: Sequence[object], texts: Mapping[int, TextEdit]
) -> dict[int, TextEdit]:
    """Lời đã sửa còn áp được: dòng vẫn mang đúng chữ lúc sửa.

    Phụ đề đã dịch lại thì chữ dòng đó khác — bản dịch mới có thể đã đúng, đè lời sửa cũ
    lên là làm hỏng nó.

    Args:
        events: Dòng phụ đề vừa nạp (có ``text``).
        texts: Lời đã sửa đã lưu.

    Returns:
        Các lời sửa áp được, theo vị trí.
    """
    return {
        position: edit
        for position, edit in texts.items()
        if 0 <= position < len(events)
        and edit.edited != edit.original
        and str(getattr(events[position], "text", "") or "") == edit.original
    }


def encode_line_edits(edits: LineEdits) -> dict[str, object]:
    """Chỉnh sửa → dict lưu được bằng JSON (khoá vị trí là chuỗi)."""
    encoded: dict[str, object] = {
        _KEY_SPEAKERS: {
            str(position): speaker for position, speaker in sorted(edits.speakers.items())
        },
        _KEY_TEXTS: {
            str(position): [edit.original, edit.edited]
            for position, edit in sorted(edits.texts.items())
        },
    }
    if edits.lines is not None:
        encoded[_KEY_LINES] = [
            [line.origin, line.start_sec, line.end_sec, line.text] for line in edits.lines
        ]
    return encoded


def decode_line_edits(raw: object) -> LineEdits:
    """Đọc lại kết quả :func:`encode_line_edits`; phần hỏng bị bỏ qua.

    Args:
        raw: Dict đã đọc từ JSON (giá trị khác dict cho kết quả rỗng).

    Returns:
        Chỉnh sửa đọc được.
    """
    if not isinstance(raw, Mapping):
        return LineEdits()
    speakers: dict[int, str] = {}
    raw_speakers = raw.get(_KEY_SPEAKERS)
    if isinstance(raw_speakers, Mapping):
        for key, speaker in raw_speakers.items():
            position = _position(key)
            if position is not None and isinstance(speaker, str):
                speakers[position] = speaker
    texts: dict[int, TextEdit] = {}
    raw_texts = raw.get(_KEY_TEXTS)
    if isinstance(raw_texts, Mapping):
        for key, pair in raw_texts.items():
            position = _position(key)
            if (
                position is not None
                and isinstance(pair, list | tuple)
                and len(pair) == 2
                and all(isinstance(part, str) for part in pair)
            ):
                texts[position] = TextEdit(original=pair[0], edited=pair[1])
    return LineEdits(speakers=speakers, texts=texts, lines=_decode_lines(raw.get(_KEY_LINES)))


def _decode_lines(raw: object) -> tuple[LineSnapshot, ...] | None:
    """Danh sách dòng đã lưu; hỏng một phần tử thì bỏ cả danh sách (không áp nửa vời)."""
    if not isinstance(raw, list | tuple):
        return None
    lines: list[LineSnapshot] = []
    for item in raw:
        try:
            origin, start, end, text = item
            line = LineSnapshot(int(origin), float(start), float(end), str(text))
        except (TypeError, ValueError):
            return None
        if line.start_sec < 0 or line.end_sec < line.start_sec:
            return None
        lines.append(line)
    return tuple(lines)


def remember_line_edits(
    store: Mapping[str, object],
    fingerprint: str,
    edits: LineEdits,
    max_subtitles: int = MAX_REMEMBERED_SUBTITLES,
) -> dict[str, object]:
    """Kho chỉnh sửa sau khi ghi chỉnh sửa của một phụ đề.

    Phụ đề vừa ghi dời xuống cuối (mới nhất); hết chỉnh sửa thì xoá khỏi kho; giữ tối đa
    ``max_subtitles`` phụ đề gần nhất để cài đặt không phình mãi.

    Args:
        store: Kho hiện tại ``{dấu phụ đề: chỉnh sửa đã mã hoá}``.
        fingerprint: Dấu của phụ đề (:func:`subtitle_fingerprint`).
        edits: Chỉnh sửa hiện tại của phụ đề đó.
        max_subtitles: Số phụ đề tối đa giữ lại.

    Returns:
        Kho mới (không sửa ``store``).
    """
    if max_subtitles <= 0:
        return {}
    kept = {key: value for key, value in store.items() if key != fingerprint}
    if fingerprint and not edits.is_empty:
        kept[fingerprint] = encode_line_edits(edits)
    return dict(list(kept.items())[-max_subtitles:])


def _position(key: object) -> int | None:
    """Khoá vị trí ("12") → 12; khoá hỏng → ``None``."""
    try:
        position = int(str(key))
    except ValueError:
        return None
    return position if position >= 0 else None


__all__ = [
    "MAX_REMEMBERED_SUBTITLES",
    "SHARED_VOICE_SPEAKER",
    "LineEdits",
    "LineSnapshot",
    "LineSpeaker",
    "TextEdit",
    "apply_speaker_overrides",
    "decode_line_edits",
    "describe_line_speakers",
    "encode_line_edits",
    "remember_line_edits",
    "restorable_text_edits",
    "subtitle_fingerprint",
]
