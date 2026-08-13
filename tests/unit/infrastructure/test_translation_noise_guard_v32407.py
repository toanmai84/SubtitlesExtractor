"""[v3.23.107] Test bảo vệ dòng "rác" OCR (không CJK) khỏi việc gửi model dịch.

Rác OCR (vd "akas") khiến model GỘP/BỎ -> lệch dòng dây chuyền (Tập 6 thực tế). Phải GIỮ
NGUYÊN VĂN dòng rác và KHÔNG gửi cho model; các dòng thật vẫn dịch và căn đúng index.
"""

from __future__ import annotations

from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
)


def _line(idx: int, text: str) -> TranslationLine:
    return TranslationLine(
        index=idx, start_ms=idx * 1000, end_ms=idx * 1000 + 900, text=text
    )


def _ctx(source_lang: str = "zh") -> TranslationContext:
    return TranslationContext(target_lang="Vietnamese", source_lang=source_lang)


# ---------- _noise_line_indices ----------

class TestNoiseLineIndices:
    def test_flags_non_cjk_lines_in_cjk_source(self) -> None:
        lines = [_line(1, "沈将军"), _line(2, "akas"),
                 _line(3, "有人泄露军中机密"), _line(4, "aks")]
        noise = GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("zh"))
        assert noise == {2, 4}

    def test_keeps_cjk_lines(self) -> None:
        lines = [_line(1, "你好"), _line(2, "世界")]
        assert GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("zh")) == set()

    def test_mixed_line_with_some_cjk_not_noise(self) -> None:
        # Dòng có CJK lẫn Latin -> vẫn là dòng thật (không phải rác).
        lines = [_line(1, "OK 你好")]
        assert GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("zh")) == set()

    def test_blank_not_flagged(self) -> None:
        lines = [_line(1, "   ")]
        assert GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("zh")) == set()

    def test_latin_source_disabled(self) -> None:
        # Nguồn Latin (en) -> KHÔNG lọc (mọi dòng đều không CJK).
        lines = [_line(1, "hello"), _line(2, "world")]
        assert GeminiSubtitleTranslator._noise_line_indices(lines, _ctx("en")) == set()

    def test_japanese_korean_sources(self) -> None:
        assert GeminiSubtitleTranslator._noise_line_indices(
            [_line(1, "xyz")], _ctx("ja")) == {1}
        assert GeminiSubtitleTranslator._noise_line_indices(
            [_line(1, "안녕")], _ctx("ko")) == set()


# ---------- Tích hợp: translate_stage giữ nguyên rác, căn đúng index ----------

class TestTranslateStagePassthrough:
    def _adapter(self, monkeypatch: Any) -> GeminiSubtitleTranslator:
        adapter = GeminiSubtitleTranslator.__new__(
            GeminiSubtitleTranslator)
        monkeypatch.setattr(adapter, "_ensure_available", lambda: None, raising=False)
        monkeypatch.setattr(
            adapter, "_response_schema_for", lambda *a, **k: {}, raising=False)
        monkeypatch.setattr(
            adapter, "_system_instruction", lambda *a, **k: "", raising=False)
        monkeypatch.setattr(
            adapter, "_build_config", lambda *a, **k: None, raising=False)
        monkeypatch.setattr(
            adapter, "_context_window", lambda *a, **k: ([], []), raising=False
        )

        # Giả lập dịch: viết hoa text, GIỮ index (mô phỏng model ngoan).
        from dataclasses import replace as _replace

        def _fake_batch(*, batch, **_kwargs):  # type: ignore[no-untyped-def]
            return [_replace(ln, text=ln.text.upper()) for ln in batch]

        monkeypatch.setattr(
            adapter, "_translate_single_batch", _fake_batch, raising=False)
        return adapter

    def test_noise_kept_verbatim_real_lines_translated(self, monkeypatch: Any) -> None:
        from subtitles_extractor.domain.ports.subtitle_translator_port import (
            TranslationStageConfig,
            TranslationStageKind,
        )

        adapter = self._adapter(monkeypatch)
        lines = [
            _line(1, "你好"), _line(2, "akas"), _line(3, "世界"),
            _line(4, "aks"), _line(5, "再见"),
        ]
        stage = TranslationStageConfig(
            kind=TranslationStageKind.LITERAL, model_name="m", batch_size=10,
        )
        out = adapter.translate_stage(
            stage=stage, context=_ctx("zh"), source_lines=lines, input_lines=lines,
            has_prior_translation=False,
        )
        by_idx = {ln.index: ln.text for ln in out}
        # Dòng thật -> "dịch" (viết hoa); dòng rác -> GIỮ NGUYÊN; căn đúng index.
        assert by_idx[1] == "你好".upper()
        assert by_idx[3] == "世界".upper()
        assert by_idx[5] == "再见".upper()
        assert by_idx[2] == "akas"  # rác giữ nguyên
        assert by_idx[4] == "aks"
        assert [ln.index for ln in out] == [1, 2, 3, 4, 5]  # đủ & đúng thứ tự
