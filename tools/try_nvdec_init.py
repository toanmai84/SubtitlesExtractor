"""Try initialize NVDEC decoders (SimpleDecoder and CreateDecoder) and print diagnostics.

Usage:
  python tools/try_nvdec_init.py --video path/to/video.mp4 --gpu 0

If --video omitted the script will search "test data" folder under repo root.
"""
from __future__ import annotations
import sys
import os
import time
import traceback
import argparse

# Ensure repo root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def discover_test_video() -> str | None:
    td = os.path.join(ROOT, 'test data')
    if os.path.isdir(td):
        for root, _dirs, files in os.walk(td):
            for f in files:
                if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.ts')):
                    return os.path.join(root, f)
            break
    # fallback top-level
    for f in os.listdir(ROOT):
        if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.ts')):
            return os.path.join(ROOT, f)
    return None


def try_simple_decoder(video: str, gpu_id: int) -> None:
    print('\n== SimpleDecoder test ==')
    try:
        from PyNvVideoCodec import SimpleDecoder
    except Exception:
        print('PyNvVideoCodec.SimpleDecoder import failed:')
        traceback.print_exc()
        return
    try:
        t0 = time.perf_counter()
        dec = SimpleDecoder(video, int(gpu_id))
        t1 = time.perf_counter()
        print(f'SimpleDecoder init OK (time {(t1-t0)*1000:.1f} ms)')
        try:
            meta = dec.get_stream_metadata()
            print('metadata:', getattr(meta, 'width', None), 'x', getattr(meta, 'height', None), 'fps=', getattr(meta, 'average_fps', None))
        except Exception:
            print('get_stream_metadata() failed:')
            traceback.print_exc()
        try:
            t0 = time.perf_counter()
            batch = dec.get_batch_frames(8)
            t1 = time.perf_counter()
            print(f'get_batch_frames(8) returned {len(batch)} frames (time {(t1-t0)*1000:.1f} ms)')
            if batch:
                fr = batch[0]
                try:
                    pts = fr.getPTS()
                except Exception:
                    pts = None
                print('sample frame PTS:', pts)
        except Exception:
            print('get_batch_frames failed:')
            traceback.print_exc()
    except Exception:
        print('SimpleDecoder init failed:')
        traceback.print_exc()


def try_create_decoder(video: str, gpu_id: int) -> None:
    print('\n== CreateDecoder/Core Decoder test ==')
    try:
        from PyNvVideoCodec import CreateDecoder
    except Exception:
        print('PyNvVideoCodec.CreateDecoder import failed:')
        traceback.print_exc()
        return
    try:
        # We don't know codec enum easily; CreateDecoder may accept probe via video
        t0 = time.perf_counter()
        # Call CreateDecoder with only gpuid to use default codec enum
        dec = CreateDecoder(gpuid=int(gpu_id))
        t1 = time.perf_counter()
        print(f'CreateDecoder object created (time {(t1-t0)*1000:.1f} ms)')
    except Exception:
        print('CreateDecoder instantiation failed (non-fatal):')
        traceback.print_exc()


def main() -> None:
    p = argparse.ArgumentParser(description='Try NVDEC init')
    p.add_argument('--video', '-v', required=False, help='Path to video')
    p.add_argument('--gpu', type=int, default=0, help='GPU id')
    args = p.parse_args()

    video = args.video or discover_test_video()
    if not video or not os.path.isfile(video):
        print('No video found. Provide --video path or place one under "test data"')
        return
    print('Using video:', video)
    print('Python:', sys.executable)

    # try SimpleDecoder and CreateDecoder
    try_simple_decoder(video, args.gpu)
    try_create_decoder(video, args.gpu)


if __name__ == '__main__':
    main()
