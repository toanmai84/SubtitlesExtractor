"""[v3.23.404] Test: nguồn CJK ngắn (thành ngữ 4 chữ) không còn báo length_anomaly giả."""

from __future__ import annotations

from types import SimpleNamespace as NS

from subtitles_extractor.application.services.translation_diagnostics import (
    detect_quality_flags,
)


def _flags(pairs: list[tuple[str, str]]) -> list[int]:
    src = [NS(index=i, text=s) for i, (s, _) in enumerate(pairs)]
    dst = [NS(index=i, text=d) for i, (_, d) in enumerate(pairs)]
    return [a["index"] for a in detect_quality_flags(src, dst)["length_anomaly"]]


def test_four_char_idiom_not_flagged() -> None:
    """Thành ngữ 4 chữ nở 2.5–3.3× là ĐÚNG — không được báo bất thường."""
    idioms = [
        ("赴汤蹈火", "dẫu phải vượt qua nước sôi lửa bỏng"),  # ratio ~3.0
        ("危在旦夕", "tình thế ngàn cân treo sợi tóc."),
        ("满门抄斩", "cả nhà họ Trịnh chu di tam tộc."),
        ("分秒必争", "phải tranh thủ từng giây từng phút"),
    ]
    assert _flags(idioms) == []


def test_genuine_over_translation_still_flagged() -> None:
    """Nguồn ngắn nhưng dịch DÀI BẤT THƯỜNG (>3.6×) vẫn phải bị bắt."""
    # 4 ký tự CJK (eff=12) với bản dịch cực dài (ratio ~5) — bất thường thật.
    pairs = [("你好世界", "x" * 60)]
    assert _flags(pairs) == [0]


def test_long_source_keeps_strict_threshold() -> None:
    """Nguồn DÀI (eff>=18) giữ ngưỡng chặt 2.5× — bắt dịch phình."""
    # 8 ký tự CJK (eff=24), dịch dài ~66 ký tự → ratio 2.75 > 2.5 → phải flag.
    pairs = [("这是一个测试句子", "y" * 66)]
    assert _flags(pairs) == [0]
