"""Worker subprocess chạy edge-tts độc lập — cách ly GPL khỏi tiến trình chính.

[v3.23.268] edge-tts là **GPL v3**. Để dùng trong sản phẩm thương mại đóng mà không lan
GPL sang toàn ứng dụng, ta chạy edge-tts trong TIẾN TRÌNH RIÊNG (giống whisperx): tiến
trình chính chỉ giao tiếp qua dòng lệnh + file WAV, KHÔNG import edge-tts. Nhờ đó edge-tts
là "công cụ ngoài" gọi qua ranh giới tiến trình, không liên kết tĩnh vào app.

Xem docs/LICENSE_ANALYSIS.md.

Cách chạy (do edge_tts_adapter gọi, không chạy tay):
    python edge_tts_subprocess.py --text "..." --voice vi-VN-... --rate "+0%" \\
        --output /tmp/out.wav

Giao thức:
- Đầu vào: tham số dòng lệnh (text, voice, rate, output path).
- Đầu ra: file WAV tại ``--output``; mã thoát 0 nếu OK, khác 0 nếu lỗi.
- Thông báo tiến độ/lỗi in ra stderr (tiến trình chính đọc để log).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path


def _eprint(message: str) -> None:
    """In ra stderr, đảm bảo đọc được chữ có dấu.

    [v3.23.342] Log thực tế hiện ``ch\\u01b0a c?i edge-tts`` — chữ bị hỏng vì stderr của
    tiến trình con trên Windows dùng bảng mã cp1252 mặc định. Ép UTF-8 để thông điệp
    chẩn đoán đọc được.
    """
    """In ra stderr để tiến trình chính đọc (stdout dành cho dữ liệu nếu cần)."""
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass
    print(message, file=sys.stderr, flush=True)


async def _synthesize(text: str, voice: str, rate: str, output_path: str) -> int:
    """Gọi edge-tts sinh audio, ghi WAV ra ``output_path``. Trả mã thoát."""
    try:
        import edge_tts
    except ImportError:
        _eprint("EDGE_TTS_MISSING: chưa cài edge-tts trong môi trường subprocess.")
        return 3

    # edge-tts xuất MP3 → lưu tạm rồi chuyển sang WAV (định dạng tiến trình chính đọc).
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    Path(mp3_path).unlink(missing_ok=True)  # chỉ giữ tên; edge-tts tự tạo
    import os

    os.close(mp3_fd)
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        try:
            # Timeout cứng: WebSocket tới Microsoft có thể treo vô hạn khi mạng lỗi.
            await asyncio.wait_for(communicate.save(mp3_path), timeout=30.0)
        except TimeoutError:
            _eprint("EDGE_TTS_TIMEOUT: quá 30s, bỏ qua dòng này.")
            return 4
        except Exception as api_exc:
            _eprint(f"EDGE_TTS_API_ERROR: {api_exc}")
            return 5

        # Chuyển MP3 → WAV bằng soundfile (BSD) / pydub (MIT) — không dùng thư viện GPL.
        if not _mp3_to_wav(mp3_path, output_path):
            return 6
        return 0
    finally:
        Path(mp3_path).unlink(missing_ok=True)


def _mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """Đọc MP3 rồi ghi WAV float32. True nếu thành công."""
    import numpy as np

    try:
        import soundfile as sf

        audio, sr = sf.read(mp3_path, dtype="float32")
    except Exception:
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_mp3(mp3_path)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            if seg.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            audio = samples
        except ImportError:
            _eprint("EDGE_TTS_NO_DECODER: thiếu soundfile/pydub để đọc MP3.")
            return False
        except Exception as dec_exc:
            _eprint(f"EDGE_TTS_DECODE_ERROR: {dec_exc}")
            return False

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    try:
        import soundfile as sf

        sf.write(wav_path, np.asarray(audio, dtype=np.float32), int(sr), subtype="FLOAT")
    except Exception as write_exc:
        _eprint(f"EDGE_TTS_WRITE_ERROR: {write_exc}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Edge TTS subprocess worker")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    return asyncio.run(_synthesize(args.text, args.voice, args.rate, args.output))


if __name__ == "__main__":
    sys.exit(main())
