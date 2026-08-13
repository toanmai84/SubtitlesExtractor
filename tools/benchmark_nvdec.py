"""Benchmark NVDEC vs PyAV vs OpenCV frame seek performance.

Usage:
    python tools/benchmark_nvdec.py --video path/to/video.mp4 --requests 50

The script will measure average/median/percentile latencies for fetching
single frames at evenly spaced timestamps using:
  - VideoProcessor with NVDEC enabled (if available)
  - VideoProcessor with NVDEC disabled (PyAV path)
  - Direct OpenCV seek/read

Results are printed to stdout.
"""
from __future__ import annotations

import argparse
import time
import statistics
import numpy as np
from typing import List
import os
import sys

# Ensure repository root is on sys.path so `from src...` imports work when
# running this script directly from tools/ directory.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.video_processor import VideoProcessor
from src.core.app_settings import app_settings
import cv2


def sample_timestamps(duration: float, n: int) -> List[float]:
    if n <= 0:
        return []
    # avoid very beginning/end to skip container edge cases
    margin = max(0.5, duration * 0.01)
    start = margin
    end = max(margin, duration - margin)
    if end <= start:
        return [0.0] * n
    return [start + (end - start) * i / max(1, n - 1) for i in range(n)]


def time_video_processor(video_path: str, timestamps: List[float]) -> List[float]:
    vp = VideoProcessor(video_path)
    # Ensure cache off
    vp.clear_frame_cache()
    timings = []
    for t in timestamps:
        t0 = time.perf_counter()
        _ = vp.lay_frame_tai_giay(t)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    vp.giai_phong()
    return timings


def time_opencv(video_path: str, timestamps: List[float]) -> List[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for OpenCV benchmark")
    timings = []
    for t in timestamps:
        t0 = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    cap.release()
    return timings


def summarize(times_ms: List[float]) -> dict:
    arr = np.array(times_ms)
    return {
        'count': int(len(arr)),
        'mean_ms': float(arr.mean()) if arr.size else 0.0,
        'median_ms': float(np.median(arr)) if arr.size else 0.0,
        'p90_ms': float(np.percentile(arr, 90)) if arr.size else 0.0,
        'p95_ms': float(np.percentile(arr, 95)) if arr.size else 0.0,
        'min_ms': float(arr.min()) if arr.size else 0.0,
        'max_ms': float(arr.max()) if arr.size else 0.0,
    }


def print_summary(name: str, stats: dict) -> None:
    print(f"--- {name} ---")
    print(f"count: {stats['count']}")
    print(f"mean:  {stats['mean_ms']:.2f} ms")
    print(f"median:{stats['median_ms']:.2f} ms")
    print(f"p90:   {stats['p90_ms']:.2f} ms")
    print(f"p95:   {stats['p95_ms']:.2f} ms")
    print(f"min:   {stats['min_ms']:.2f} ms")
    print(f"max:   {stats['max_ms']:.2f} ms")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark NVDEC vs PyAV vs OpenCV frame seek")
    p.add_argument('--video', '-v', required=False, help='Path to video file (optional). If omitted script will search "test data" folder.')
    p.add_argument('--requests', '-n', type=int, default=50, help='Number of timestamp requests')
    args = p.parse_args()

    # If video not provided, try to find a test video under repo "test data" folder
    if not args.video:
        test_videos = []
        td = os.path.join(ROOT, "test data")
        if os.path.isdir(td):
            for rootd, _dirs, files in os.walk(td):
                for f in files:
                    if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.ts')):
                        test_videos.append(os.path.join(rootd, f))
                break
        # fallback: look in repo root top-level files
        if not test_videos:
            for f in os.listdir(ROOT):
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.ts')):
                    test_videos.append(os.path.join(ROOT, f))
        if test_videos:
            args.video = test_videos[0]
            print(f"No --video provided — using discovered test video: {args.video}")
        else:
            print("Error: --video not provided and no test videos found under 'test data' or repo root.")
            return

    # probe duration using OpenCV
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0.0
    cap.release()

    if duration <= 0:
        print("Cannot determine video duration — abort")
        return

    timestamps = sample_timestamps(duration, args.requests)
    print(f"Benchmarking {args.video} — duration={duration:.2f}s — {len(timestamps)} requests")

    # 1) NVDEC enabled
    app_settings.set('hw/nvdec_enabled', True)
    app_settings.set('hw/device', app_settings.get('hw/device', 0))
    print('Running VideoProcessor with NVDEC enabled (if available) ...')
    t_nv = time_video_processor(args.video, timestamps)
    stats_nv = summarize(t_nv)
    print_summary('VideoProcessor (NVDEC enabled)', stats_nv)

    # 2) NVDEC disabled (force PyAV/OpenCV path)
    app_settings.set('hw/nvdec_enabled', False)
    print('Running VideoProcessor with NVDEC disabled (PyAV path) ...')
    t_pyav = time_video_processor(args.video, timestamps)
    stats_pyav = summarize(t_pyav)
    print_summary('VideoProcessor (NVDEC disabled / PyAV)', stats_pyav)

    # 3) Direct OpenCV
    print('Running direct OpenCV seek/read ...')
    t_cv = time_opencv(args.video, timestamps)
    stats_cv = summarize(t_cv)
    print_summary('OpenCV seek/read', stats_cv)

    # 4) Batch decode benchmarks using NVDEC/Core
    print('Running batch decode benchmarks (targeted timestamps) ...')
    # Prepare batch targets (reuse same timestamps list)
    try:
        from src.core.nvdecoder import decode_frames_nvdec, decode_frames_core, decode_frames_core_ctx
        # NVDEC sequential (SimpleDecoder)
        try:
            t0 = time.perf_counter()
            count = 0
            for ts, _ in decode_frames_nvdec(args.video, timestamps, gpu_id=app_settings.get('hw/device', 0)):
                count += 1
            total_ms = (time.perf_counter() - t0) * 1000.0
            stats = {'count': count, 'total_ms': total_ms, 'mean_ms_per_frame': (total_ms / count) if count else 0.0}
            print('--- decode_frames_nvdec (SimpleDecoder sequential) ---')
            print(f"count: {stats['count']}, total: {stats['total_ms']:.2f} ms, mean/frame: {stats['mean_ms_per_frame']:.2f} ms")
        except Exception as exc:
            print('decode_frames_nvdec failed or not available:', exc)

        # Core decoder (PyAV demux -> NVDEC) using decode_frames_core_ctx
        try:
            t0 = time.perf_counter()
            count = 0
            with decode_frames_core_ctx(args.video, gpu_id=app_settings.get('hw/device', 0), target_timestamps=timestamps) as gen:
                for _ in gen:
                    count += 1
            total_ms = (time.perf_counter() - t0) * 1000.0
            stats = {'count': count, 'total_ms': total_ms, 'mean_ms_per_frame': (total_ms / count) if count else 0.0}
            print('--- decode_frames_core (PyAV demux -> NVDEC) ---')
            print(f"count: {stats['count']}, total: {stats['total_ms']:.2f} ms, mean/frame: {stats['mean_ms_per_frame']:.2f} ms")
        except Exception as exc:
            print('decode_frames_core failed or not available:', exc)
    except Exception:
        print('NVDEC batch decode functions not available in this environment.')

    print('Done')


if __name__ == '__main__':
    main()
