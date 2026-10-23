"""Test [v3.23.25] EdgeTTS + audio_mastering có timeout chống treo vô hạn."""

from __future__ import annotations

import inspect

from subtitles_extractor.infrastructure.tts import audio_mastering, edge_tts_adapter
from subtitles_extractor.infrastructure.tts.edge_tts_adapter import EdgeTTSAdapter


def test_async_generate_wraps_save_in_timeout() -> None:
    # [v3.23.268] edge-tts cách ly GPL vào subprocess. Timeout chống treo nay nằm ở:
    # (1) worker _synthesize (wait_for 30s quanh communicate.save), (2)
    # _run_edge_subprocess (wait_for 35s quanh proc.communicate). Kiểm cả hai.
    from subtitles_extractor.infrastructure.tts import edge_tts_subprocess

    worker_src = inspect.getsource(edge_tts_subprocess)
    assert "wait_for" in worker_src
    assert "timeout=30.0" in worker_src
    assert "TimeoutError" in worker_src

    run_src = inspect.getsource(EdgeTTSAdapter._run_edge_subprocess)
    assert "wait_for" in run_src
    assert "timeout=35.0" in run_src


def test_edge_tts_cach_ly_gpl_subprocess() -> None:
    """[v3.23.268] Cách ly GPL: đường chạy MẶC ĐỊNH là subprocess, không import edge_tts.

    [v3.23.547] Chính sách đã đổi ở v3.23.328 — khi bản đóng gói KHÔNG tìm được Python
    ngoài, adapter lùi về chạy edge-tts ngay trong tiến trình (``_run_edge_in_process``),
    và tệp spec đã gom ``edge_tts`` vào bundle. Bất biến còn hiệu lực vì thế hẹp lại,
    nhưng vẫn là bất biến thật: edge_tts chỉ được import LƯỜI, bên trong đúng hàm dự
    phòng đó — không bao giờ ở cấp module hay ở đường chạy chính. Nhờ vậy phiên làm việc
    dùng subprocess (mặc định) không nạp một dòng mã GPL nào.

    Hệ quả cấp phép của thay đổi v3.23.328 đã được ghi lại ở ``docs/LICENSE_ANALYSIS.md``.
    """
    module_src = inspect.getsource(edge_tts_adapter)
    for line in module_src.splitlines():
        # Import cấp module (không thụt lề) là điều TUYỆT ĐỐI cấm: nạp mã GPL ngay khi
        # ứng dụng khởi động, bất kể người dùng có dùng Edge TTS hay không.
        if line.startswith(("import edge_tts", "from edge_tts")):
            raise AssertionError("Không được import edge_tts ở cấp module")

    fallback_src = inspect.getsource(EdgeTTSAdapter._run_edge_in_process)
    assert "import edge_tts" in fallback_src, (
        "Import phải nằm trong hàm dự phòng để chỉ nạp khi thật sự cần"
    )

    # Ngoài hàm dự phòng, không phương thức nào khác của adapter được import edge_tts.
    for name, member in vars(EdgeTTSAdapter).items():
        if name == "_run_edge_in_process":
            continue
        target = getattr(member, "__func__", member)
        if not callable(target) or isinstance(member, (str, bytes)):
            continue
        try:
            member_src = inspect.getsource(target)
        except (TypeError, OSError):
            continue
        for line in member_src.splitlines():
            assert not line.strip().startswith(("import edge_tts", "from edge_tts")), (
                f"{name} không được import edge_tts — chỉ đường dự phòng mới được"
            )


def test_audio_mastering_encode_has_timeout() -> None:
    # ffmpeg encode output phải có timeout + bắt TimeoutExpired.
    src = inspect.getsource(audio_mastering)
    assert "timeout=600" in src
    assert "TimeoutExpired" in src
