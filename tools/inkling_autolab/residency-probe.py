"""Read-only Windows working-set probe using one task-owned synthetic expert.

Separates mapped-page validity from a warm buffered file read; never flushes
system caches, trims other processes, or changes working-set limits.
"""
import ctypes as C
import json
import msvcrt
import time
from pathlib import Path

p = Path('C:/Users/devcloud/inkling-autolab/synthetic-experts/synthetic_0.bin')
k = C.WinDLL('kernel32', use_last_error=True)
ps = C.WinDLL('psapi', use_last_error=True)
k.GetCurrentProcess.restype = C.c_void_p
k.CreateFileMappingW.argtypes = [C.c_void_p, C.c_void_p, C.c_ulong, C.c_ulong, C.c_ulong, C.c_wchar_p]
k.CreateFileMappingW.restype = C.c_void_p
k.MapViewOfFile.argtypes = [C.c_void_p, C.c_ulong, C.c_ulong, C.c_ulong, C.c_size_t]
k.MapViewOfFile.restype = C.c_void_p
k.UnmapViewOfFile.argtypes = [C.c_void_p]
k.CloseHandle.argtypes = [C.c_void_p]
class Entry(C.Structure):
    _fields_ = [('address', C.c_void_p), ('flags', C.c_size_t)]
class Range(C.Structure):
    _fields_ = [('address', C.c_void_p), ('size', C.c_size_t)]
ps.QueryWorkingSetEx.argtypes = [C.c_void_p, C.c_void_p, C.c_ulong]
k.PrefetchVirtualMemory.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(Range), C.c_ulong]
process = k.GetCurrentProcess()
size = p.stat().st_size
with p.open('rb') as f:
    mapping = k.CreateFileMappingW(msvcrt.get_osfhandle(f.fileno()), None, 2, 0, 0, None)
    if not mapping:
        raise C.WinError(C.get_last_error())
    base = k.MapViewOfFile(mapping, 4, 0, 0, 0)
    if not base:
        k.CloseHandle(mapping)
        raise C.WinError(C.get_last_error())
    try:
        pages = (size + 4095)//4096
        offsets = list(range(0, pages, max(1, pages//64)))[:64]
        def query(phase, elapsed_ms=None):
            entries = (Entry * len(offsets))(*[Entry(base + i*4096, 0) for i in offsets])
            if not ps.QueryWorkingSetEx(process, entries, C.sizeof(entries)):
                raise C.WinError(C.get_last_error())
            records.append({'phase': phase, 'valid': sum(e.flags & 1 for e in entries),
                            'sampled': len(entries), 'elapsed_ms': elapsed_ms})
        records = []
        query('new_mapping')
        for i in range(3):
            start = time.perf_counter()
            data = p.read_bytes()
            ms = (time.perf_counter()-start)*1000
            assert len(data) == size
            del data
            query(f'buffered_read_{i}', ms)
        r = Range(base, size)
        ok = k.PrefetchVirtualMemory(process, 1, C.byref(r), 0)
        query(f'prefetch_returned_{ok}')
        start = time.perf_counter()
        checksum = sum(C.c_ubyte.from_address(base+i*4096).value for i in range(pages))
        query('after_mapping_page_walk', (time.perf_counter()-start)*1000)
        print(json.dumps({'scope': 'warm_file_cache_working_set_probe', 'bin_bytes': size,
                          'page_walk_checksum': checksum, 'phases': records}, indent=2))
    finally:
        k.UnmapViewOfFile(base)
        k.CloseHandle(mapping)
