"""[v3.23.157] Test chuỗi filter nén video: hwupload_cuda bền + fps đặt ĐẦU chuỗi.

Log The Hot Spot/Three Against the World (stderr đầy đủ nhờ v155): tầng GPU-full gãy
đàm phán format ("Error reinitializing filters" / -40) khi decoder trả frame software
-> scale_cuda (chỉ nhận CUDA) không nối được. Fix chuẩn Jellyfin: ``hwupload_cuda``
trước ``scale_cuda`` (frame CUDA đi xuyên, frame software được đẩy lên GPU). Kèm tối
ưu: ``fps`` đặt ĐẦU mọi chuỗi -> chỉ scale số khung thật cần (~1/24 công scale).
"""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    _vf_cpu_scale,
    _vf_gpu_full,
    _vf_gpu_upload,
)


def test_gpu_full_chain_pure_hw_no_upload() -> None:
    # [v3.23.159] Tầng thuần-GPU KHÔNG chèn hwupload_cuda: bản legacy của filter này
    # không cho frame CUDA đi xuyên (danh sách format vào không có 'cuda') -> chèn
    # vào làm gãy cả đường decode-GPU tốt. fps vẫn đứng đầu (tiết kiệm ~24x scale).
    chain = _vf_gpu_full(360, "1")
    parts = chain.split(",")
    assert parts[0] == "fps=1"
    assert parts[1] == "scale_cuda=w=-2:h=360:format=nv12"
    # [v3.23.167] hwdownload,format=nv12 kéo frame VRAM->RAM sau scale_cuda.
    assert "hwdownload" in chain and chain.endswith("format=nv12")
    assert "hwupload" not in chain


def test_gpu_upload_chain_for_software_frames() -> None:
    # Tầng 1.25: frame software (decode NVDEC về RAM) -> upload -> scale GPU.
    chain = _vf_gpu_upload(360, "1")
    parts = chain.split(",")
    assert parts[0] == "fps=1"
    assert parts[1] == "hwupload_cuda"
    assert parts[2] == "scale_cuda=w=-2:h=360:format=nv12"
    assert "hwdownload" in chain and chain.endswith("format=nv12")


def test_cpu_scale_chain_fps_first() -> None:
    chain = _vf_cpu_scale(360, "1")
    parts = chain.split(",")
    assert parts[0] == "fps=1"
    assert parts[1] == "scale=-2:360"
    assert parts[2] == "format=yuv420p"


def test_chains_respect_parameters() -> None:
    assert "h=540" in _vf_gpu_full(540, "2")
    assert "fps=2" in _vf_gpu_full(540, "2")
    assert "h=540" in _vf_gpu_upload(540, "2")
    assert "scale=-2:720" in _vf_cpu_scale(720, "0.5")
