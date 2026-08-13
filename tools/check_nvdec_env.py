r"""Diagnostic script to inspect NVDEC / PyNvVideoCodec environment on Windows.

Run from workspace root in PowerShell:
  r"python tools\check_nvdec_env.py"

It prints:
 - Python executable and bitness
 - site-packages locations
 - existence and path of PyNvVideoCodec package and _PyNvVideoCodec.* file
 - locations of any cudart64_*.dll found under site-packages, CUDA_PATH, and PATH
 - tries to ctypes.CDLL the found cudart and reports errors
 - attempts import PyNvVideoCodec and prints exception
"""
from __future__ import annotations
import sys, os, glob, ctypes, traceback

def main():
    print('Python:', sys.executable, sys.version)
    print('Platform:', sys.platform, sys.maxsize > 2**32 and '64-bit' or '32-bit')
    print()

    import site
    paths = []
    try:
        up = site.getusersitepackages()
        if up:
            paths.append(up)
    except Exception:
        pass
    try:
        sp = site.getsitepackages()
        for p in sp:
            paths.append(p)
    except Exception:
        pass
    print('Site paths to probe:')
    for p in paths:
        print('  ', p)
    print()

    # Find PyNvVideoCodec
    found = False
    for p in paths:
        candidate = os.path.join(p, 'PyNvVideoCodec')
        if os.path.isdir(candidate):
            print('Found PyNvVideoCodec dir:', candidate)
            found = True
            print('  files:')
            for f in os.listdir(candidate):
                print('   ', f)
    if not found:
        print('PyNvVideoCodec package dir not found in site paths')
    print()

    # Locate _PyNvVideoCodec.* under site-packages
    pyd_matches = []
    for p in paths:
        pyd_matches.extend(glob.glob(os.path.join(p, '**', '_PyNvVideoCodec.*'), recursive=True))
    if pyd_matches:
        print('Found native module candidates:')
        for m in pyd_matches:
            print('  ', m)
    else:
        print('No _PyNvVideoCodec.* found under site-packages')
    print()

    # Search for cudart DLLs
    cudarts = []
    for p in paths:
        cudarts.extend(glob.glob(os.path.join(p, '**', 'cudart64_*.dll'), recursive=True))
    # CUDA_PATH
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_PATH_V10_1')
    if cuda_path:
        cudarts.extend(glob.glob(os.path.join(cuda_path, 'bin', 'cudart64_*.dll')))
    # PATH
    for p in os.environ.get('PATH', '').split(os.pathsep):
        try:
            cudarts.extend(glob.glob(os.path.join(p, 'cudart64_*.dll')))
        except Exception:
            pass
    cudarts = sorted(set(cudarts))
    if cudarts:
        print('Found cudart DLLs:')
        for c in cudarts:
            print('  ', c)
    else:
        print('No cudart64_*.dll found in site-packages/CUDA_PATH/PATH')
    print()

    # Try to load first cudart
    if cudarts:
        p = cudarts[0]
        print('Try loading:', p)
        try:
            lib = ctypes.CDLL(p)
            print('  ctypes.CDLL OK')
            try:
                print('  has cudaMemcpy:', hasattr(lib, 'cudaMemcpy'))
            except Exception:
                pass
        except Exception:
            print('  ctypes.CDLL failed:')
            traceback.print_exc()
    print()

    # Try import PyNvVideoCodec to show exception
    print('Attempting "import PyNvVideoCodec"...')
    try:
        import PyNvVideoCodec
        print('import PyNvVideoCodec OK, location:', getattr(PyNvVideoCodec, '__file__', 'n/a'))
    except Exception as e:
        print('Import failed:')
        traceback.print_exc()

if __name__ == '__main__':
    main()
