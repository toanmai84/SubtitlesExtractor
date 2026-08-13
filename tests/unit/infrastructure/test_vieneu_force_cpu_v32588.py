"""[v3.23.188] Test ép CPU/ONNX + thông báo lỗi cuDNN cho VieNeu-TTS.

Lỗi thực tế từ máy người dùng: ``[WinError 127] ... cudnn_engines_precompiled64_9.dll``
— VieNeu SDK tự chuyển sang PyTorch khi thấy CUDA, gặp bộ cuDNN lệch bản -> lỗi tải DLL.
Fix: ép CPU/ONNX (ẩn GPU qua ``CUDA_VISIBLE_DEVICES``) + thông báo lỗi tiếng Việt rõ ràng
có hướng dẫn khắc phục. Test này khoá hai hàm thuần đó.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.tts.vieneu_tts_adapter import (
    VieNeuTtsAdapter,
    describe_voice_error,
    resolve_device_env,
)

# ── resolve_device_env ───────────────────────────────────────────────────


def test_force_cpu_hides_gpu() -> None:
    env = resolve_device_env(force_cpu=True)
    assert env == {"CUDA_VISIBLE_DEVICES": ""}


def test_no_force_cpu_empty_env() -> None:
    assert resolve_device_env(force_cpu=False) == {}


# ── describe_voice_error ─────────────────────────────────────────────────


def test_dll_error_suggests_force_cpu_when_off() -> None:
    error = OSError(
        "[WinError 127] The specified procedure could not be found. "
        "Error loading cudnn_engines_precompiled64_9.dll"
    )
    message = describe_voice_error(error, force_cpu=False)
    assert "Ép chạy CPU" in message
    assert "cuDNN" in message or "CUDA" in message


def test_dll_error_suggests_onnx_when_already_cpu() -> None:
    error = OSError("[WinError 127] cudnn_engines_precompiled64_9.dll could not be found")
    message = describe_voice_error(error, force_cpu=True)
    assert "onnxruntime" in message.lower()


def test_generic_error_plain_message() -> None:
    error = ValueError("voice not found")
    message = describe_voice_error(error, force_cpu=True)
    assert "voice not found" in message
    assert "cuDNN" not in message


def test_cudnn_keyword_detected() -> None:
    error = RuntimeError("Failed to load cuDNN backend")
    message = describe_voice_error(error, force_cpu=False)
    assert "Ép chạy CPU" in message


# ── Adapter nhận force_cpu ────────────────────────────────────────────────


def test_adapter_defaults_force_cpu_true() -> None:
    adapter = VieNeuTtsAdapter()
    assert adapter._force_cpu is True


def test_adapter_force_cpu_configurable() -> None:
    adapter = VieNeuTtsAdapter(force_cpu=False)
    assert adapter._force_cpu is False
