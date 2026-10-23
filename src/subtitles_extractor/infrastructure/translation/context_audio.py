"""[v3.23.610] Tham số âm thanh cho đoạn video ngữ cảnh gửi Gemini (hàm thuần).

Học từ ``extract_audio_from_video`` của gemini-srt-translator, đo lại bằng ffmpeg thật
trên nguồn thử (xem ``tests/unit/infrastructure/test_context_audio_v323610.py``):

* ``aresample=async=1:first_pts=0``: lấp khoảng hở mốc thời gian trong luồng âm thanh và
  cho âm thanh bắt đầu đúng 0 s. Không có nó, sau một khoảng hở 2 s mọi âm thanh của đoạn
  sớm 2 s so với hình.
* Nguồn nhiều kênh có kênh giữa (5.1, 7.1…): lời thoại nằm ở FC. Trộn mono ưu tiên kênh
  giữa và giữ một phần hai kênh trước (nhạc nền, không khí cảnh), bỏ kênh vòm và LFE (hiệu
  ứng, tiếng ầm). Dấu ``<`` chuẩn hoá hệ số để không chạm trần. Trên nguồn thử, tỉ lệ lời/
  nền tăng ~9 dB so với ``-ac 1``. gemini-srt-translator giữ RIÊNG kênh giữa; ở đây giữ
  lại nhạc vì ứng dụng dùng âm thanh cả cho ngữ điệu và tag cảm xúc.
* Chọn luồng âm thanh MẶC ĐỊNH (cờ disposition) thay vì luôn luồng đầu tiên.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "AUDIO_RESYNC_FILTER",
    "DIALOGUE_DOWNMIX_FILTER",
    "AudioStreamInfo",
    "context_audio_args",
    "select_context_audio",
]

AUDIO_RESYNC_FILTER = "aresample=async=1:first_pts=0"
DIALOGUE_DOWNMIX_FILTER = "pan=mono|c0<FC+0.3*FL+0.3*FR"
_DIALOGUE_CHANNELS = frozenset({"FC", "FL", "FR"})
#: Mã hoá âm thanh của đoạn ngữ cảnh: mono 16 kHz đủ cho lời nói, nhẹ khi tải lên.
_ENCODE_ARGS = ("-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "64k")


@dataclass(frozen=True)
class AudioStreamInfo:
    """Một luồng âm thanh của video.

    Attributes:
        position: Thứ tự trong các luồng âm thanh (dùng cho ``-map 0:a:N``).
        channels: Số kênh.
        channel_names: Tên kênh theo bố cục (``"FL"``, ``"FC"``…); rỗng nếu không rõ.
        is_default: Luồng được đánh dấu mặc định.
    """

    position: int
    channels: int
    channel_names: tuple[str, ...] = ()
    is_default: bool = False

    @property
    def has_centre_dialogue(self) -> bool:
        """Có kênh giữa cùng hai kênh trước (bố cục vòm điển hình của phim)."""
        return self.channels >= 3 and _DIALOGUE_CHANNELS.issubset(self.channel_names)


def select_context_audio(streams: Sequence[AudioStreamInfo]) -> AudioStreamInfo | None:
    """Luồng âm thanh dùng làm ngữ cảnh: luồng mặc định, không có thì luồng đầu.

    Args:
        streams: Các luồng âm thanh theo thứ tự trong tệp.

    Returns:
        Luồng được chọn; ``None`` nếu video không có âm thanh.
    """
    for stream in streams:
        if stream.is_default:
            return stream
    return streams[0] if streams else None


def context_audio_args(audio: AudioStreamInfo | None) -> list[str]:
    """Tham số ffmpeg chọn, lọc và mã hoá âm thanh của đoạn ngữ cảnh.

    Args:
        audio: Luồng đã chọn; ``None`` = không đọc được thông tin (dùng luồng đầu,
            trộn mono mặc định).

    Returns:
        Danh sách tham số (``-map``, ``-af``, mã hoá).
    """
    position = audio.position if audio is not None else 0
    filters = [AUDIO_RESYNC_FILTER]
    if audio is not None and audio.has_centre_dialogue:
        filters.insert(0, DIALOGUE_DOWNMIX_FILTER)
    return ["-map", f"0:a:{position}?", "-af", ",".join(filters), *_ENCODE_ARGS]
