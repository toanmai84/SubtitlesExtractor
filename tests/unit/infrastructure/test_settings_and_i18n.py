"""Test :class:`SettingsService` + :class:`JsonFileRepository` + :class:`JsonTranslator`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subtitles_extractor.domain.exceptions import ConfigurationError
from subtitles_extractor.domain.value_objects.device_kind import DeviceKind
from subtitles_extractor.infrastructure.i18n.json_translator import JsonTranslator
from subtitles_extractor.infrastructure.settings.application_settings import (
    ApplicationSettings,
)
from subtitles_extractor.infrastructure.settings.json_file_repository import (
    JsonFileRepository,
)
from subtitles_extractor.infrastructure.settings.settings_service import (
    SettingsService,
)


# ── ApplicationSettings ───────────────────────────────────────────────────


class TestApplicationSettings:
    def test_default_values_are_valid(self) -> None:
        settings = ApplicationSettings()
        assert settings.hardware.device == DeviceKind.GPU
        assert settings.hardware.batch_size_ocr == 128
        assert settings.frame.sample_step_sec == pytest.approx(0.04)

    def test_rejects_out_of_range_batch_size_ocr(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            ApplicationSettings.model_validate({"hardware": {"batch_size_ocr": 0}})

    def test_round_trip_dump_validate(self) -> None:
        original = ApplicationSettings()
        dumped = original.model_dump(mode="json")
        restored = ApplicationSettings.model_validate(dumped)
        assert restored == original


# ── JsonFileRepository ────────────────────────────────────────────────────


class TestJsonFileRepository:
    def test_load_returns_default_when_missing(self, tmp_path: Path) -> None:
        repo = JsonFileRepository(tmp_path / "empty.json")
        assert repo.load("k", "default") == "default"

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        repo = JsonFileRepository(tmp_path / "config.json")
        repo.save("device", "gpu")
        repo.save("batch", 32)
        repo.flush()

        repo2 = JsonFileRepository(tmp_path / "config.json")
        assert repo2.load("device", None) == "gpu"
        assert repo2.load("batch", None) == 32

    def test_atomic_no_partial_file(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        repo = JsonFileRepository(target)
        repo.save("k", {"nested": [1, 2, 3]})
        repo.flush()
        # Đọc lại bằng json để chắc rằng nội dung là JSON hợp lệ.
        assert json.loads(target.read_text(encoding="utf-8")) == {
            "k": {"nested": [1, 2, 3]}
        }

    def test_reset_clears_data(self, tmp_path: Path) -> None:
        repo = JsonFileRepository(tmp_path / "c.json")
        repo.save("a", 1)
        repo.flush()
        repo.reset()
        repo.flush()
        assert repo.load("a", "missing") == "missing"


# ── SettingsService ───────────────────────────────────────────────────────


class TestSettingsService:
    def test_initial_state_is_defaults(self, tmp_path: Path) -> None:
        repo = JsonFileRepository(tmp_path / "c.json")
        service = SettingsService(repo)
        assert service.current.hardware.device == DeviceKind.GPU

    def test_update_persists_to_repo(self, tmp_path: Path) -> None:
        config_file = tmp_path / "c.json"
        repo = JsonFileRepository(config_file)
        service = SettingsService(repo)
        service.update(hardware={"batch_size_ocr": 64, "device": "cpu"})
        # update() là debounce — cần flush để ghi disk ngay.
        service.flush()
        assert config_file.exists()

        # Reload từ disk — giá trị phải còn.
        repo2 = JsonFileRepository(config_file)
        service2 = SettingsService(repo2)
        assert service2.current.hardware.batch_size_ocr == 64
        assert service2.current.hardware.device == DeviceKind.CPU

    def test_update_debounces_writes(self, tmp_path: Path) -> None:
        """Nhiều update liên tục chỉ ghi đĩa 1 lần ở cuối debounce."""
        import time
        config_file = tmp_path / "c.json"
        repo = JsonFileRepository(config_file)
        service = SettingsService(repo)

        # 5 update nhanh — debounce timer luôn reset.
        for i in range(5):
            service.update(hardware={"batch_size_ocr": 16 + i})
        # Trong khoảng debounce, file chưa được ghi.
        # (300ms - chưa kịp).
        # Đợi qua debounce + chút buffer.
        time.sleep(0.5)
        assert config_file.exists()

        repo2 = JsonFileRepository(config_file)
        service2 = SettingsService(repo2)
        # Giá trị cuối cùng (i=4) → 16+4=20.
        assert service2.current.hardware.batch_size_ocr == 20

    def test_update_rolls_back_on_invalid(self, tmp_path: Path) -> None:
        repo = JsonFileRepository(tmp_path / "c.json")
        service = SettingsService(repo)
        original_batch = service.current.hardware.batch_size_ocr
        with pytest.raises(ConfigurationError):
            service.update(hardware={"batch_size_ocr": -1})  # vi phạm ge=1
        # Cấu hình hiện tại không bị thay đổi.
        assert service.current.hardware.batch_size_ocr == original_batch

    def test_reset_to_defaults(self, tmp_path: Path) -> None:
        repo = JsonFileRepository(tmp_path / "c.json")
        service = SettingsService(repo)
        service.update(hardware={"batch_size_ocr": 99})
        assert service.current.hardware.batch_size_ocr == 99
        service.reset_to_defaults()
        assert service.current.hardware.batch_size_ocr == 128


# ── JsonTranslator ────────────────────────────────────────────────────────


class TestJsonTranslator:
    @pytest.fixture
    def i18n_dir(self, tmp_path: Path) -> Path:
        data = {
            "_meta": {"language": "vi"},
            "app": {"title": "Trích xuất phụ đề"},
            "extract": {"status_done": "Xong! {count} câu trong {seconds}s"},
        }
        target = tmp_path / "strings_vi.json"
        target.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        return tmp_path

    def test_translate_simple_key(self, i18n_dir: Path) -> None:
        translator = JsonTranslator(i18n_dir)
        assert translator.translate("app.title") == "Trích xuất phụ đề"

    def test_translate_with_placeholders(self, i18n_dir: Path) -> None:
        translator = JsonTranslator(i18n_dir)
        result = translator.translate(
            "extract.status_done", count=10, seconds="3.5"
        )
        assert result == "Xong! 10 câu trong 3.5s"

    def test_translate_missing_key_returns_key(self, i18n_dir: Path) -> None:
        translator = JsonTranslator(i18n_dir)
        assert translator.translate("non.existent.key") == "non.existent.key"

    def test_set_locale_changes_lookup(self, i18n_dir: Path) -> None:
        # Tạo strings_en.json
        en_data = {"_meta": {"language": "en"}, "app": {"title": "Subtitles"}}
        (i18n_dir / "strings_en.json").write_text(
            json.dumps(en_data), encoding="utf-8"
        )
        translator = JsonTranslator(i18n_dir)
        translator.set_locale("en")
        assert translator.translate("app.title") == "Subtitles"

    def test_available_locales(self, i18n_dir: Path) -> None:
        translator = JsonTranslator(i18n_dir)
        assert "vi" in translator.available_locales()
