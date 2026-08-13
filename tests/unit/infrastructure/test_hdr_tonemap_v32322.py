"""Test [v3.23.24] phát hiện HDR/10-bit để uỷ quyền PyAV (đúng màu, không nhân đôi)."""

from __future__ import annotations

from subtitles_extractor.infrastructure.video.decoders.pynvvideocodec_frame_sampler import (
    PyNvVideoCodecFrameSampler,
)


def _stream(trc: str = "", prim: str = "", pix: str = "yuv420p"):
    class Ctx:
        color_trc = trc
        color_primaries = prim
        pix_fmt = pix
    class Stream:
        codec_context = Ctx()
    return Stream()


class TestDetectHdr:
    def test_pq_bt2020_10bit(self) -> None:
        assert PyNvVideoCodecFrameSampler._detect_hdr_transfer(
            _stream("smpte2084", "bt2020", "p010le")) is True

    def test_hdr_vivid_no_trc(self) -> None:
        # HDR Vivid (CUVV) đôi khi không khai color_trc — BT.2020 + 10-bit là đủ.
        assert PyNvVideoCodecFrameSampler._detect_hdr_transfer(
            _stream("", "bt2020", "yuv420p10le")) is True

    def test_hlg(self) -> None:
        assert PyNvVideoCodecFrameSampler._detect_hdr_transfer(
            _stream("arib-std-b67", "bt2020", "p010le")) is True

    def test_sdr_8bit_not_hdr(self) -> None:
        assert PyNvVideoCodecFrameSampler._detect_hdr_transfer(
            _stream("bt709", "bt709", "yuv420p")) is False

    def test_sdr_bt709_10bit_not_hdr(self) -> None:
        # 10-bit nhưng BT.709 (không BT.2020) → không HDR (sẽ đi PyAV vì 10-bit, không tonemap).
        assert PyNvVideoCodecFrameSampler._detect_hdr_transfer(
            _stream("bt709", "bt709", "yuv420p10le")) is False

    def test_none_safe(self) -> None:
        class Stream:
            codec_context = type("C", (), {"color_trc": None, "color_primaries": None, "pix_fmt": None})()
        assert PyNvVideoCodecFrameSampler._detect_hdr_transfer(Stream()) is False
