"""[v3.23.596] Lỗi lồng tiếng/thuyết minh tìm ra trên bộ chẩn đoán thật 10/10/2026.

Phim "灭门惨案II借种..AI修复版[老舅电影laody.net发布]" (1245 dòng, VieNeu, 19 vai):

* Xuất bản không tìm thấy ``<tên>.tts.vi.srt`` vì ``[...]`` trong tên phim là tập ký tự
  của ``glob`` → phim xuất ra mang phụ đề mốc GỐC, lệch giọng thuyết minh.
* Nhãn vai ngoài bảng ghi bằng chữ Hán ("老伯" — ông bác già) không đọc được giới tính →
  có thể nhận giọng nữ.
* Cùng một nhân vật ghi ba kiểu "洪震的姘頭" / "洪震之姘頭" / "洪震の姘頭" → tách thành
  hai vai, hai giọng.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subtitles_extractor.application.services.batch_publish_plan import find_tts_subtitle
from subtitles_extractor.application.services.batch_translate_plan import (
    find_existing_translation,
)
from subtitles_extractor.application.services.batch_tts_plan import (
    find_translated_subtitle,
)
from subtitles_extractor.application.services.cue_label_canonicalizer import (
    canonical_label,
    roster_label_map,
)
from subtitles_extractor.application.services.dubbing_cast import (
    AGE_CHILD,
    AGE_ELDER,
    DubbingSource,
    build_cast,
    line_speakers,
)
from subtitles_extractor.application.services.output_variants import find_output_variants
from subtitles_extractor.application.services.publish_subtitle_selector import (
    SubtitleSource,
    choose_publish_subtitle,
)
from subtitles_extractor.domain.entities.subtitle_event import SubtitleEvent
from subtitles_extractor.domain.value_objects.time_interval import TimeInterval
from subtitles_extractor.domain.value_objects.voice_profile import (
    VOICE_FEMALE,
    VOICE_MALE,
)

FILM = "灭门惨案II借种..AI修复版[老舅电影laody.net发布]"

#: Hai dòng bảng nhân vật thật (trích nguyên văn, cắt phần mô tả).
ROSTER = (
    "洪震 (Hồng Chấn) [nam, ~40] — Kẻ lưu manh khét tiếng làng Đông Long.\n"
    "警員 / 阿權 / 洪志權 (Công an viên / A Quyền / Hồng Chí Quyền) [nam, ~25] — Con "
    "trai của Hồng Chấn.\n"
    "洪震的姘頭 (Nhân tình của Hồng Chấn) [nữ, ~22] — Cô gái lẳng lơ.\n"
)
#: Visual Cues thật dòng 286-289 và 431 (nhãn người nói/người nghe giữ nguyên văn).
CUES = (
    '[{"id":1,"spk":"洪震的姘頭","to":"警員 / 阿權 / 洪志權"},'
    '{"id":2,"spk":"洪震的姘頭","to":"警員 / 阿權 / 洪志權"},'
    '{"id":3,"spk":"洪震の姘頭","to":"警員 / 阿權 / 洪志權"},'
    '{"id":4,"spk":"洪震の姘頭","to":"警員 / 阿權 / 洪志權"},'
    '{"id":5,"spk":"老伯","to":"洪震"},'
    '{"id":6,"spk":"洪震","to":"洪震之姘頭"}]'
)


class TestBracketedFileNames:
    def _folder(self, tmp_path: Path) -> Path:
        for name in (f"{FILM}.mp4", f"{FILM}.flac", f"{FILM}.tts.vi.srt",
                     f"{FILM}.translate.vi.srt"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        return tmp_path

    def test_variants_are_found_despite_brackets(self, tmp_path: Path) -> None:
        folder = self._folder(tmp_path)
        expected = [folder / f"{FILM}.tts.vi.srt"]
        assert find_output_variants(folder, FILM, "tts") == expected

    def test_publish_uses_the_synced_subtitle(self, tmp_path: Path) -> None:
        folder = self._folder(tmp_path)
        translated = folder / "phu_de_da_dich.srt"
        translated.write_text("x", encoding="utf-8")
        # Dự án lưu ngôn ngữ đích dạng tên đầy đủ "Vietnamese" — không khớp mã "vi", nên
        # chỉ còn đường dò mẫu (đường hỏng vì ngoặc vuông trước bản này).
        choice = choose_publish_subtitle(
            tts_audio_path=folder / f"{FILM}.flac",
            translated_subtitle_path=translated,
            target_language="Vietnamese",
            audio_will_be_used=True,
        )
        assert choice.source is SubtitleSource.TTS_SYNCED
        assert choice.warning is None

    def test_batch_plans_find_their_files(self, tmp_path: Path) -> None:
        folder = self._folder(tmp_path)
        video = folder / f"{FILM}.mp4"
        assert find_tts_subtitle(video, "Vietnamese") == folder / f"{FILM}.tts.vi.srt"
        expected = folder / f"{FILM}.translate.vi.srt"
        assert find_existing_translation(video, "Vietnamese") == expected
        assert find_translated_subtitle(video, "Vietnamese") == expected


class TestLabelVariants:
    @pytest.mark.parametrize("label", ["洪震の姘頭", "洪震之姘頭", "洪震的姘頭"])
    def test_possessive_variants_map_to_roster(self, label: str) -> None:
        assert canonical_label(label, roster_label_map(ROSTER)) == "洪震的姘頭"

    def test_one_character_one_role(self) -> None:
        events = [
            SubtitleEvent(index=i, text=f"câu {i}", interval=TimeInterval(i, i + 0.8))
            for i in range(1, 7)
        ]
        source = DubbingSource(
            characters=ROSTER, visual_cues=CUES,
            source_spans=tuple((float(i), i + 0.8) for i in range(1, 7)),
        )
        cast = {member.name: member for member in build_cast(
            line_speakers(events, source), ROSTER
        )}
        assert "洪震の姘頭" not in cast
        assert cast["洪震的姘頭"].line_count == 4
        assert cast["洪震的姘頭"].gender == VOICE_FEMALE


class TestCjkLabelTraits:
    @pytest.mark.parametrize(("label", "gender", "age"), [
        ("老伯", VOICE_MALE, AGE_ELDER),       # nhãn thật: cue "Ông cụ chê bai…"
        ("大嫂", VOICE_FEMALE, ""),
        ("女婿", VOICE_MALE, ""),              # con rể: không bị chữ "女" bắt nhầm
        ("小女孩", VOICE_FEMALE, AGE_CHILD),
        ("老太婆", VOICE_FEMALE, AGE_ELDER),
        ("Lão nông", VOICE_MALE, AGE_ELDER),  # nhãn tiếng Việt giữ cách đọc cũ
        ("Người bán hàng", "", ""),
        ("阿積", "", ""),
    ])
    def test_traits(self, label: str, gender: str, age: str) -> None:
        member = build_cast([label], "")[0]
        assert (member.gender, member.age_group) == (gender, age)
