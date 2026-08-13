"""[v3.23.122] Test: parse bảng quota + export/replace/default của quota manager."""

from __future__ import annotations

from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
    GeminiQuotaManager,
)
from subtitles_extractor.presentation.pages.translate_page import parse_quota_rows


def test_parse_valid_rows() -> None:
    rows = [("Gemini-3.5-Flash", "5", "250000", "20"), ("custom-x", "10", "1000", "100")]
    limits, errors = parse_quota_rows(rows)
    assert errors == []
    assert limits["gemini-3.5-flash"] == {"rpm": 5, "tpm": 250000, "rpd": 20}
    assert limits["custom-x"]["rpd"] == 100


def test_parse_skips_empty_model() -> None:
    limits, errors = parse_quota_rows([("", "5", "5", "5"), ("  ", "1", "1", "1")])
    assert limits == {}
    assert errors == []


def test_parse_reports_non_integer() -> None:
    limits, errors = parse_quota_rows([("m", "abc", "5", "5")])
    assert limits == {}
    assert errors and "số nguyên" in errors[0]


def test_parse_reports_non_positive() -> None:
    limits, errors = parse_quota_rows([("m", "0", "5", "5")])
    assert limits == {}
    assert errors and "> 0" in errors[0]


def test_manager_export_replace_roundtrip() -> None:
    mgr = GeminiQuotaManager()
    mgr.replace_limits_dict({"my-model": {"rpm": 7, "tpm": 999, "rpd": 42}})
    exported = mgr.export_limits_dict()
    assert exported == {"my-model": {"rpm": 7, "tpm": 999, "rpd": 42}}
    # Replace bằng tập mới -> tập cũ bị thay hoàn toàn.
    mgr.replace_limits_dict({"other": {"rpm": 1, "tpm": 2, "rpd": 3}})
    assert "my-model" not in mgr.export_limits_dict()


def test_replace_ignores_malformed_entries() -> None:
    mgr = GeminiQuotaManager()
    mgr.replace_limits_dict({
        "good": {"rpm": 1, "tpm": 2, "rpd": 3},
        "bad": {"rpm": "x", "tpm": 2, "rpd": 3},
        "": {"rpm": 1, "tpm": 2, "rpd": 3},
    })
    exported = mgr.export_limits_dict()
    assert "good" in exported and "bad" not in exported and "" not in exported


def test_custom_limit_overrides_default_in_acquire() -> None:
    mgr = GeminiQuotaManager()
    mgr.replace_limits_dict({"gemini-x": {"rpm": 100, "tpm": 10**9, "rpd": 1}})
    assert mgr.get_remaining("gemini-x", key_id="k")["rpd_limit"] == 1


def test_default_tier_limits_exposed() -> None:
    defaults = GeminiQuotaManager.default_tier_limits()
    assert "flash" in defaults
    assert set(defaults["flash"]) == {"rpm", "tpm", "rpd"}


# ── v3.23.124: fingerprint dùng chung, snapshot, swap nguyên tử ───────────


def test_key_fingerprint_matches_adapter() -> None:
    from subtitles_extractor.infrastructure.translation import (
        gemini_translation_adapter as adp,
    )
    key = "AIzaSomeSecretKey123"
    adapter_fp = adp.GeminiSubtitleTranslator._fingerprint(key)
    assert GeminiQuotaManager.key_fingerprint(key) == adapter_fp
    assert GeminiQuotaManager.key_fingerprint("") == ""


def test_snapshot_reports_per_key_usage() -> None:
    from subtitles_extractor.infrastructure.translation.gemini_quota_manager import (
        RateLimit,
    )
    mgr = GeminiQuotaManager(rate_limits={"m": RateLimit(rpm=100, tpm=10**9, rpd=20)})
    mgr.acquire("m", 1, key_id="A")
    mgr.acquire("m", 1, key_id="A")
    mgr.acquire("m", 1, key_id="B")
    snap = {(s["key_id"], s["model"]): s for s in mgr.snapshot()}
    assert snap[("A", "m")]["rpd_used"] == 2
    assert snap[("A", "m")]["rpd_remaining"] == 18
    assert snap[("B", "m")]["rpd_used"] == 1


def test_replace_limits_atomic_keeps_complete_dict() -> None:
    mgr = GeminiQuotaManager()
    mgr.replace_limits_dict({"a": {"rpm": 1, "tpm": 2, "rpd": 3}})
    # Sau khi thay bằng tập mới, model cũ không còn override -> suy theo tiền tố.
    mgr.replace_limits_dict({"gemini-x": {"rpm": 9, "tpm": 9, "rpd": 9}})
    assert "a" not in mgr.export_limits_dict()
    assert mgr.export_limits_dict()["gemini-x"]["rpd"] == 9
