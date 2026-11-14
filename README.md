# Subtitles Extractor — v3.23.726

**English** | [Tiếng Việt](README.vi.md)

A desktop application for **extracting hardcoded or embedded subtitles, translating them with AI, and producing dubbed audio**. It is built with Python and PySide6 using **Clean Architecture / Hexagonal Architecture**.

The complete workflow in one app: **Extract → Edit → Translate → Dub**.

## Features

### Subtitle extraction
- Supports MP4, MKV, AVI, MOV, FLV, and WEBM input; exports SRT and ASS.
- **Hard-sub OCR** with PaddleOCR, including NVIDIA GPU acceleration when available and CPU fallback.
- **Speech recognition (STT)** with WhisperX for videos without hardcoded subtitles.
- Extracts subtitle tracks embedded in video files.
- Smart frame sampling with time intervals, perceptual-hash deduplication, and pixel comparison.
- Greedy and Viterbi subtitle grouping algorithms.
- Automatic subtitle-region (ROI) detection; batch processing can validate and reuse a previous episode's ROI when appropriate.
- Experimental/optional NVIDIA GPU path with NVDEC and CuPy for video decoding and image preprocessing on the GPU; falls back to PyAV/CPU when GPU components are unavailable.
- OCR result voting counts independent OCR reads rather than duplicated frames, reducing bias when identical frames are reused.
- Handles CJK (Chinese, Japanese, and Korean) and Latin text throughout the pipeline.
- Flags suspicious OCR lines for AI review and improves OCR safety when re-OCR or settings changes overlap with extraction.
- Filters moving scene objects that can otherwise be mistaken for subtitles.

### Subtitle editing
- Open and edit `.srt` and `.ass` files; insert, delete, split, merge, and edit subtitle lines in a table.
- Re-run OCR on an individual line or video frame.
- Undo and redo, audio waveform editing, and DAW-style keyboard shortcuts.
- Synchronized video preview with MPV, PyAV, or OpenCV.

### AI translation
- Multi-stage translation: **preprocessing → draft translation → style refinement → localization**.
- Choose **Google Gemini** or a custom provider with an OpenAI-compatible API. Configure its API URL and key, then load the provider's available text models in the Translate page.
- Gemini model selection includes supported free and paid text-generation models, with quota guidance in the model catalog.
- Global context analysis can identify languages and characters and create plot summaries and glossaries.
- Video-based visual cues help the AI understand scene context, forms of address, and speakers when using Gemini.
- If a Gemini model reaches its daily quota, the app can automatically switch to the configured fallback model for translation and analysis, when fallback is enabled and available.
- Translation stages are cached and can be reviewed or edited individually; unchanged stages can be reused on the next run.
- Translate selected lines again, edit translations in place, compare stages, and check terminology consistency.
- Translation Memory shares translated phrases, glossaries, and character information across episodes in a series.
- Features that require video or audio analysis need Google Gemini; custom text-only providers can still be used for supported translation and text tasks.

### Series workflow
- Process episodes in sequence through extraction, translation, and publishing.
- Review results between stages or stop the workflow after translation.
- Validate and reuse the subtitle region from a previous episode when it matches, reducing repeated full-frame searches.
- Batch merging of adjacent subtitle fragments avoids repeated full-table refreshes on long projects.

### Text-to-speech (TTS)
- Generate synchronized speech with Edge TTS, Gemini TTS, and VieNeu.
- Adjust timing and time-stretch audio to fit subtitle durations.
- Detect and handle repeated speech in short lines; normalize loudness using sample-rate-aware LUFS measurement.
- Edit line-specific addressee and dubbing details, regenerate individual lines, and manage recoverable audio history.
- Gemini TTS can optionally use per-line emotion/style cues; this option is off by default.
- For series, saved character voice assignments can be reused across episodes, including when character names have known aliases.

### Settings, diagnostics, and recovery
- Runtime settings use `pydantic-settings` and can be overridden with environment variables such as `SE_HW_*`, `SE_OCR_*`, `SE_VIDCTX_*`, and `SE_TRANS_*`.
- Settings are saved automatically with QSettings. The interface supports Vietnamese and light, dark, and automatic themes.
- The Windows build bundles CuPy's required Python modules while excluding large CUDA DLLs; CUDA runtime components are loaded separately when GPU acceleration is enabled.
- Safer MPV playback and throttled seeking help reduce Windows playback crashes; the app can fall back to another video preview backend when needed.
- Native crash logs help diagnose unexpected exits.
- SQLite databases are backed up and can be recovered when corruption is detected.

## Architecture

```text
src/subtitles_extractor/
├── domain/          # Entities, value objects, and ports; pure Python
├── application/     # Use cases and application services
├── infrastructure/  # OCR, STT, translation, TTS, video, database, and subtitle adapters
├── presentation/    # Qt UI, view models, workers, and widgets
└── composition/     # Dependency injection and application bootstrap
```

Dependency direction: `presentation → application → domain ← infrastructure`. The domain does not depend on Qt, PaddleOCR, or OpenCV; layers communicate through ports (`Protocol`).

## Installation and running

### Requirements
- Python **3.11 or later** (Python 3.12 recommended).
- Optional NVIDIA GPU and CUDA for accelerated OCR/STT.
- A Google Gemini API key for Gemini features, or an API key from a supported OpenAI-compatible provider for text-only AI tasks.

### Install

```bash
git clone https://github.com/toanmai84/SubtitlesExtractor.git
cd SubtitlesExtractor
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and run the app:

```bash
pip install -r requirements.txt
python main.py
```

`main.py` adds `src/` to the Python path automatically.

For development tools:

```bash
pip install -r requirements-dev.txt
```

Run unit tests:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit -q
```

## Quick guide: translate a multi-episode series

1. Put the episodes of a series in the same folder.
2. Extract subtitles from the first episode with OCR or STT, then review them in **Edit**.
3. In **Translate**, configure an AI provider and analyze the subtitles to generate context, a glossary, and character information.
4. Run the desired translation stages. Refine individual lines, compare stages, check terminology, or edit a stage before translating again.
5. Process subsequent episodes in the same series to reuse translation memory and character information.
6. Optionally use **TTS** to dub the translated subtitles.

## Environment variable examples

```bash
export SE_HW_DEVICE=cpu
export SE_HW_USE_TENSORRT=true
export SE_OCR_SCORE_THRESHOLD=0.5
export SE_POST_USE_VITERBI=true
export SE_TRANS_DEFAULT_BATCH_SIZE=50
export SE_VIDCTX_FPS=1.0
python main.py
```

## Contributing

- Follow PEP 8, Black (90-character line length), and Ruff.
- Add type hints and catch specific exceptions.
- Use English identifiers; user-facing UI text and logs follow the project's localization conventions.
- Include tests with code changes and run the relevant pytest suite before packaging.

## License

MIT. See [LICENSE](LICENSE).
