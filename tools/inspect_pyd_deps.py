r"""Inspect native dependencies of PyNvVideoCodec .pyd on Windows.

Usage:
  python tools\inspect_pyd_deps.py

This script will:
 - locate PyNvVideoCodec_*.pyd under site-packages
 - list imported DLL names (via pefile if available)
 - try to locate each DLL on disk (PATH, CUDA_PATH, site-packages)
 - optionally try ctypes.CDLL(load) each discovered candidate and report success/failure

Prints a summary of missing / found dependencies to help diagnose the "DLL not found" error.
"""
from __future__ import annotations
import os
import sys
import glob
import ctypes
from typing import List

try:
    import pefile
except Exception:
    pefile = None

import site


def find_pyd_candidates() -> List[str]:
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
    candidates = []
    for p in paths:
        candidates.extend(glob.glob(os.path.join(p, '**', 'PyNvVideoCodec_*.pyd'), recursive=True))
    return sorted(set(candidates))


def get_imports_from_pyd(pyd_path: str) -> List[str]:
    if pefile is None:
        # fallback: try to parse via pefile not available
        print('pefile not installed — please pip install pefile to get full import list')
        return []
    try:
        pe = pefile.PE(pyd_path)
        imports = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') and pe.DIRECTORY_ENTRY_IMPORT:
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                name = entry.dll.decode('utf-8', errors='ignore') if isinstance(entry.dll, bytes) else str(entry.dll)
                imports.append(name)
        return imports
    except Exception as e:
        print('Error parsing PE:', e)
        return []


def locate_dll_on_system(dll_name: str) -> List[str]:
    candidates = []
    # search PATH
    for p in os.environ.get('PATH', '').split(os.pathsep):
        try:
            full = os.path.join(p, dll_name)
            if os.path.isfile(full):
                candidates.append(full)
        except Exception:
            pass
    # search CUDA_PATH
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_PATH_V10_1')
    if cuda_path:
        cand = os.path.join(cuda_path, 'bin', dll_name)
        if os.path.isfile(cand):
            candidates.append(cand)
    # search site-packages (common locations)
    try:
        up = site.getusersitepackages()
        if up:
            candidates.extend(glob.glob(os.path.join(up, '**', dll_name), recursive=True))
    except Exception:
        pass
    try:
        for sp in site.getsitepackages():
            candidates.extend(glob.glob(os.path.join(sp, '**', dll_name), recursive=True))
    except Exception:
        pass
    # Windows system32
    sys32 = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'System32')
    cand = os.path.join(sys32, dll_name)
    if os.path.isfile(cand):
        candidates.append(cand)
    return sorted(set(candidates))


def try_load(path_or_name: str) -> bool:
    try:
        ctypes.CDLL(path_or_name)
        return True
    except Exception:
        return False


def main():
    print('Python:', sys.executable)
    pyds = find_pyd_candidates()
    if not pyds:
        print('No PyNvVideoCodec_*.pyd found in site-packages. Install PyNvVideoCodec into this env.')
        return
    print('Found pyd candidates:')
    for i,p in enumerate(pyds):
        print(f'  [{i}] {p}')
    pyd = pyds[0]
    print('\nUsing:', pyd)

    imports = get_imports_from_pyd(pyd)
    if not imports:
        print('No import list available (pefile missing or parse failed).')
        print('You can still try ctypes.CDLL on the pyd to see the detailed error:')
        try:
            ctypes.CDLL(pyd)
            print('ctypes.CDLL on pyd succeeded (surprising).')
        except Exception as e:
            print('ctypes.CDLL failed:')
            import traceback
            traceback.print_exc()
        return

    print('\nImported DLLs:')
    for dll in imports:
        print('  ', dll)

    print('\nLocating each imported DLL on filesystem (PATH, CUDA_PATH, site-packages, System32)...')
    missing = []
    for dll in imports:
        found = locate_dll_on_system(dll)
        if found:
            print(f'\n{dll} -> found:')
            for f in found:
                ok = try_load(f)
                print(f'    {f}  (loadable={ok})')
        else:
            print(f'\n{dll} -> NOT FOUND on system')
            missing.append(dll)

    if missing:
        print('\nMissing DLLs that were not found on system:')
        for m in missing:
            print('  ', m)
    else:
        print('\nAll imported DLL names were located somewhere on disk. If import still fails, a dependency of one of these DLLs may be missing (use Dependencies tool).')

if __name__ == '__main__':
    main()
