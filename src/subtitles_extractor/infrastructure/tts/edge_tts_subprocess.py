"""Worker subprocess chạy edge-tts độc lập — cách ly GPL khỏi tiến trình chính.

[v3.23.268] edge-tts là **GPL v3**. Để dùng trong sản phẩm thương mại đóng mà không lan
GPL sang toàn ứng dụng, ta chạy edge-tts trong TIẾN TRÌNH RIÊNG (giống whisperx): tiến
trình chính chỉ giao tiếp qua dòng lệnh + file WAV, KHÔNG import edge-tts. Nhờ đó edge-tts
là "công cụ ngoài" gọi qua ranh giới tiến trình, không liên kết tĩnh vào app.

Xem docs/LICENSE_ANALYSIS.md.

Cách chạy (do edge_tts_adapter gọi, không chạy tay):
    python edge_tts_subprocess.py --text="..." --voice=vi-VN-... --rate="-10%" \\
        --pitch="+0Hz" --output=/tmp/out.wav

Giao thức:
- Đầu vào: tham số dòng lệnh (text, voice, rate, pitch, output path).
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


#: Máy chủ dịch vụ đọc của Edge (để xét danh sách bỏ qua proxy ``no_proxy``).
EDGE_TTS_HOST = "speech.platform.bing.com"


def resolve_system_proxy() -> str | None:
    """[v3.23.673] Proxy hệ thống cho kết nối WebSocket của edge-tts.

    edge-tts dùng aiohttp, mà aiohttp KHÔNG tự đọc ``HTTPS_PROXY`` / proxy của Windows: chạy
    thật sau proxy, mọi câu đều "audio rỗng" (thực chất ``ClientConnectorDNSError`` tới
    speech.platform.bing.com) rồi báo "kiểm tra mạng" dù trình duyệt vẫn ra mạng bình thường.
    Lấy proxy như ``urllib`` (biến môi trường; trên Windows cả cài đặt hệ thống), chỉ nhận
    proxy HTTP (aiohttp không hỗ trợ SOCKS) và tôn trọng ``no_proxy``.

    Returns:
        URL proxy, hoặc ``None`` khi không có / không dùng được.
    """
    import urllib.request

    try:
        proxies = urllib.request.getproxies()
        if urllib.request.proxy_bypass(EDGE_TTS_HOST):
            return None
    except Exception:  # đọc cài đặt hệ thống lỗi thì đi thẳng như cũ
        return None
    for key in ("https", "http", "all"):
        value = str(proxies.get(key) or "").strip()
        if not value:
            continue
        if "://" not in value:
            value = f"http://{value}"
        if value.lower().startswith("http://") or value.lower().startswith("https://"):
            return value
    return None


async def _synthesize(
    text: str, voice: str, rate: str, output_path: str, pitch: str = "+0Hz"
) -> int:
    """Gọi edge-tts sinh audio, ghi WAV ra ``output_path``. Trả mã thoát.

    ``pitch`` [v3.23.593]: cao độ tương đối — biến thể giọng khi lồng tiếng nhiều vai.
    """
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
        communicate = edge_tts.Communicate(
            text, voice, rate=rate, pitch=pitch, proxy=resolve_system_proxy()
        )
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


def build_parser() -> argparse.ArgumentParser:
    """Bộ đọc tham số dòng lệnh của worker (dùng chung cho test giao thức với adapter).

    Adapter truyền mọi giá trị dạng ``--khoá=giá_trị`` vì giá trị có thể bắt đầu bằng "-"
    (tốc độ ``-10%``, cao độ ``-12Hz``, câu thoại ``-Đi!``).
    """
    parser = argparse.ArgumentParser(description="Edge TTS subprocess worker")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--pitch", default="+0Hz")
    parser.add_argument("--output", required=True)
    return parser


def _hide_child_consoles() -> None:
    """[v3.23.709] Worker chạy không có console: tiến trình con của nó (ffmpeg, lệnh dò của thư viện…) sẽ bật cửa sổ
    cmd nếu không ẩn. Bản rút gọn của ``hidden_process.install_hidden_console_default`` (worker không import gói
    ứng dụng)."""
    import subprocess
    import sys

    if sys.platform != "win32" or getattr(subprocess.Popen, "_se_hidden_console_default", False):
        return
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    original_init = subprocess.Popen.__init__

    def hidden_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if flag and not kwargs.get("creationflags") and kwargs.get("startupinfo") is None:
            kwargs["creationflags"] = flag
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = hidden_init  # type: ignore[method-assign]
    subprocess.Popen._se_hidden_console_default = True  # type: ignore[attr-defined]


def main() -> int:
    _hide_child_consoles()
    args = build_parser().parse_args()

    return asyncio.run(
        _synthesize(args.text, args.voice, args.rate, args.output, args.pitch)
    )


if __name__ == "__main__":
    sys.exit(main())
