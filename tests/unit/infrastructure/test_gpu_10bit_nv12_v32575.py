"""[v3.23.175] Test 2 sửa lỗi từ log Mario Galaxy 2160p DV (10-bit HDR).

1. Filter GPU: scale_cuda phải ép format=nv12 TRÊN GPU trước hwdownload. Video 10-bit
   HDR/DV cho frame CUDA p010le; hwdownload thẳng sang nv12 -> "Invalid output format
   nv12 for hwframe download" (-22). Phải chuyển p010->nv12 trên GPU rồi mới tải RAM.
2. Worker: phân biệt "người dùng huỷ THẬT" vs TranslationCancelledError phát sinh từ
   timeout/lỗi mạng kéo dài -> không báo nhầm "đã huỷ" che giấu lỗi thật.
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    _vf_gpu_full,
    _vf_gpu_upload,
)


def test_gpu_full_scales_to_nv12_on_gpu_before_download() -> None:
    chain = _vf_gpu_full(360, "1")
    # scale_cuda phải mang :format=nv12 (chuyển p010->nv12 TRÊN GPU cho video 10-bit).
    assert "scale_cuda=w=-2:h=360:format=nv12" in chain
    # Thứ tự: scale_cuda (có format) đứng TRƯỚC hwdownload.
    assert chain.index("scale_cuda") < chain.index("hwdownload")
    assert chain.endswith("hwdownload,format=nv12")


def test_gpu_upload_scales_to_nv12_on_gpu_before_download() -> None:
    chain = _vf_gpu_upload(360, "1")
    assert "scale_cuda=w=-2:h=360:format=nv12" in chain
    assert chain.index("hwupload_cuda") < chain.index("scale_cuda")
    assert chain.index("scale_cuda") < chain.index("hwdownload")


def test_both_chains_end_with_ram_download() -> None:
    # Cả hai tầng kết thúc bằng hwdownload,format=nv12 -> frame về RAM cho encoder.
    for chain in (_vf_gpu_full(540, "2"), _vf_gpu_upload(540, "2")):
        assert chain.endswith("hwdownload,format=nv12")
        # format=nv12 xuất hiện ĐÚNG HAI lần: trong scale_cuda và sau hwdownload.
        assert chain.count("format=nv12") == 2


# ── Phân loại cancel thật vs gián đoạn lỗi (hàm thuần, không cần Qt) ──────

from subtitles_extractor.presentation.workers.cancellation_outcome import (  # noqa: E402
    CancellationOutcome,
    classify_cancellation_outcome,
)


def test_user_cancel_when_flag_set() -> None:
    assert (
        classify_cancellation_outcome(cancel_flag_set=True)
        is CancellationOutcome.USER_CANCELLED
    )


def test_interrupted_by_error_when_flag_not_set() -> None:
    # TranslationCancelledError phát sinh nhưng cờ huỷ KHÔNG bật -> lỗi mạng/timeout,
    # KHÔNG phải người dùng huỷ -> báo lỗi để thử lại (không che giấu).
    assert (
        classify_cancellation_outcome(cancel_flag_set=False)
        is CancellationOutcome.INTERRUPTED_BY_ERROR
    )
