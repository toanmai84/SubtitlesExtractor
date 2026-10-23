"""[v3.23.605] Ref video đã tải lại bằng key mới được dùng lại ở MỌI giai đoạn sau.

Nhật ký thật 11/10 21:10 (灭门惨案II, 17 đoạn video): key #1 hết hạn mức
'gemini-3.7-flash' lúc 20:36:54 ở cuối giai đoạn dịch thô → xoay sang key #2. Sau đó:

* Mỗi lô của giai đoạn 3 VÀ giai đoạn 4 đều dò ``files.get`` ref của key cũ → 403, mỗi
  ref bị dò HAI lần ("Handle video 'files/pj94770dl0gp' cần tải lại" lặp đôi), rồi gọi
  tải lại.
* Giai đoạn 4 lặp lại y nguyên vòng 403 + tải lại cho đủ 17 đoạn dù giai đoạn 3 vừa tải
  lại xong: mỗi giai đoạn nhận lại ``list(request.video_refs)`` — ref GỐC của key cũ.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from subtitles_extractor.domain.ports.subtitle_translator_port import (
    TranslationContext,
    TranslationLine,
    TranslationStageConfig,
    TranslationStageKind,
)
from subtitles_extractor.infrastructure.translation.gemini_translation_adapter import (
    GeminiSubtitleTranslator,
    healed_ref_key,
    substitute_healed_refs,
)
from subtitles_extractor.infrastructure.translation.gemini_video_context import (
    RemoteVideoRef,
)

SEGMENTS = 4
SEGMENT_SEC = 10.0


def _refs(prefix: str) -> list[RemoteVideoRef]:
    return [
        RemoteVideoRef(
            idx,
            f"files/{prefix}{idx}",
            idx * SEGMENT_SEC,
            (idx + 1) * SEGMENT_SEC,
            "ACTIVE",
        )
        for idx in range(SEGMENTS)
    ]


def _lines() -> list[TranslationLine]:
    # Hai dòng mỗi đoạn, lô 2 dòng → mỗi lô phủ đúng một đoạn video.
    return [
        TranslationLine(
            index=i + 1,
            start_ms=int(i * SEGMENT_SEC * 500) + 100,
            end_ms=int(i * SEGMENT_SEC * 500) + 900,
            text=f"句{i}",
            original_text=f"句{i}",
        )
        for i in range(SEGMENTS * 2)
    ]


class _RotatedAdapter:
    """Adapter vừa xoay từ KEY1 sang KEY2: file 'OLD' (của KEY1) luôn 403."""

    def __init__(self) -> None:
        self.adapter = GeminiSubtitleTranslator(api_key="KEY1,KEY2")
        self.adapter._switch_to_key(1)
        self.probed_old: list[str] = []
        self.reuploads: list[list[tuple[float, float]]] = []
        self.sent: list[list[str]] = []

        def _resolve(refs: list[Any] | None) -> list[Any]:
            handles = []
            for ref in refs or []:
                if "NEW" in ref.remote_name:
                    handles.append(SimpleNamespace(name=ref.remote_name))
                else:
                    self.probed_old.append(ref.remote_name)
            return handles

        def _reupload(_key: str, ranges: list[tuple[float, float]]) -> list[Any]:
            self.reuploads.append(list(ranges))
            wanted = set(ranges)
            return [r for r in _refs("NEW") if (r.start_sec, r.end_sec) in wanted]

        def _single_batch(*, batch, video_files, **_kwargs):
            self.sent.append([h.name for h in video_files or []])
            return [replace(line, text=f"dịch {line.index}") for line in batch]

        self.adapter._resolve_video_handles = _resolve  # type: ignore[assignment]
        self.adapter._video_reupload_cb = _reupload
        self.adapter._ensure_available = lambda: None  # type: ignore[assignment]
        self.adapter._translate_single_batch = _single_batch  # type: ignore[assignment]

    def run_stage(self, kind: TranslationStageKind) -> None:
        lines = _lines()
        self.adapter.translate_stage(
            stage=TranslationStageConfig(kind=kind, model_name="m", batch_size=2),
            context=TranslationContext(target_lang="Vietnamese"),
            source_lines=lines,
            input_lines=lines,
            has_prior_translation=kind is not TranslationStageKind.LITERAL,
            # Giống use case: mỗi giai đoạn nhận lại ref GỐC (của key cũ).
            video_refs=_refs("OLD"),
            attach_video=True,
        )


class TestStageAfterRotation:
    def test_each_old_ref_is_probed_at_most_once(self) -> None:
        harness = _RotatedAdapter()
        harness.run_stage(TranslationStageKind.STYLE)
        assert sorted(harness.probed_old) == sorted(set(harness.probed_old))
        assert len(harness.reuploads) == SEGMENTS
        assert harness.sent == [[f"files/NEW{i}"] for i in range(SEGMENTS)]

    def test_next_stage_reuses_healed_refs_without_403_or_reupload(self) -> None:
        harness = _RotatedAdapter()
        harness.run_stage(TranslationStageKind.STYLE)
        probes_before = len(harness.probed_old)
        uploads_before = len(harness.reuploads)
        harness.sent.clear()

        harness.run_stage(TranslationStageKind.LOCALIZE)

        assert len(harness.probed_old) == probes_before
        assert len(harness.reuploads) == uploads_before
        assert harness.sent == [[f"files/NEW{i}"] for i in range(SEGMENTS)]

    def test_healed_refs_are_per_key(self) -> None:
        harness = _RotatedAdapter()
        harness.run_stage(TranslationStageKind.STYLE)
        harness.adapter._switch_to_key(0)
        uploads_before = len(harness.reuploads)
        harness.run_stage(TranslationStageKind.LOCALIZE)
        # Ref tải lại bằng KEY2 không dùng được cho KEY1 → phải tải lại (callback giả).
        assert len(harness.reuploads) > uploads_before


class TestSubstituteHealedRefs:
    def test_replaces_only_ranges_healed_under_that_key(self) -> None:
        fresh = _refs("NEW")
        healed = {healed_ref_key("K2", fresh[1]): fresh[1]}
        refs = _refs("OLD")
        assert substitute_healed_refs(refs, healed, "K2") == 1
        assert [r.remote_name for r in refs] == [
            "files/OLD0", "files/NEW1", "files/OLD2", "files/OLD3",
        ]
        other = _refs("OLD")
        assert substitute_healed_refs(other, healed, "K1") == 0
