// elastic_win.cpp — the ramlab elastic-allocator posture for Windows.
//
// The Linux path preloads an interposer via LD_PRELOAD. Windows has no
// LD_PRELOAD, so we inline-hook the UCRT allocation family with Microsoft
// Detours: patching `ucrtbase!malloc` (and friends) in place catches every
// caller in the process — including OpenVINO/oneDNN DLLs loaded *after* the
// hook installs — because they all resolve those symbols to the same
// `ucrtbase` code. Large allocations are then served from a section backed by
// a real temp file (CreateFileMapping over a DELETE_ON_CLOSE file), so their
// dirty pages are written to the file and reclaimed from the working set under
// pressure instead of consuming private commit — the Windows analog of the
// Linux MAP_SHARED/O_TMPFILE trick, and the same D-015 mechanism.
//
// Installed from `elastic_install_win`, called by cascadia BEFORE the OpenVINO
// engine loads (so DLLs bind through the hook) and while the process is quiet.
// Not installed from DllMain — Detours transactions must not run under loader
// lock.

#include <windows.h>
#include <detours.h>
#include <malloc.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

// ---- real (trampoline) pointers, initialized to the CRT entries ----
static void*  (*real_malloc)(size_t)                 = malloc;
static void   (*real_free)(void*)                    = free;
static void*  (*real_calloc)(size_t, size_t)         = calloc;
static void*  (*real_realloc)(void*, size_t)         = realloc;
static size_t (*real_msize)(void*)                   = _msize;
static void*  (*real_aligned_malloc)(size_t, size_t) = _aligned_malloc;
static void   (*real_aligned_free)(void*)            = _aligned_free;
static void*  (*real_aligned_realloc)(void*, size_t, size_t) = _aligned_realloc;
static size_t (*real_aligned_msize)(void*, size_t, size_t)   = _aligned_msize;

// ---- config ----
static size_t g_threshold = (size_t)1 << 20;     // ELASTIC_MIN_MB
static size_t g_pool_cap  = (size_t)8 << 30;     // ELASTIC_POOL_MB
static wchar_t g_dir[MAX_PATH];
static LONG64  g_big = 0, g_big_bytes = 0, g_pool_hit = 0, g_fallback = 0;

#define PAGE 4096ULL
#define MAGIC 0xE1A5710C0FFEE5ULL

typedef struct {
    uint64_t magic;
    size_t   user;      // bytes the caller asked for
    size_t   total;     // header + rounded payload, the mapped length
    HANDLE   hmap;      // section handle
    HANDLE   hfile;     // backing file handle
} hdr_t;

// ---- retention pool: freed mappings kept mapped for zero-cost reuse ----
#define POOL_SLOTS 256
typedef struct { void* base; size_t total; HANDLE hmap; HANDLE hfile; } pool_ent;
static pool_ent g_pool[POOL_SLOTS];
static size_t g_pool_bytes = 0;
static CRITICAL_SECTION g_cs;
static int g_ready = 0;

static void* pool_take(size_t total_needed) {
    if (!g_pool_cap) return NULL;
    EnterCriticalSection(&g_cs);
    int best = -1;
    for (int i = 0; i < POOL_SLOTS; i++) {
        if (!g_pool[i].base) continue;
        if (g_pool[i].total >= total_needed && g_pool[i].total <= 2 * total_needed &&
            (best < 0 || g_pool[i].total < g_pool[best].total))
            best = i;
    }
    void* base = NULL;
    if (best >= 0) {
        base = g_pool[best].base;
        g_pool_bytes -= g_pool[best].total;
        g_pool[best].base = NULL;
        InterlockedIncrement64(&g_pool_hit);
    }
    LeaveCriticalSection(&g_cs);
    return base;
}

static int pool_put(void* base, size_t total, HANDLE hmap, HANDLE hfile) {
    if (!g_pool_cap) return 0;
    EnterCriticalSection(&g_cs);
    int ok = 0;
    if (g_pool_bytes + total <= g_pool_cap) {
        for (int i = 0; i < POOL_SLOTS; i++) {
            if (!g_pool[i].base) {
                g_pool[i].base = base; g_pool[i].total = total;
                g_pool[i].hmap = hmap; g_pool[i].hfile = hfile;
                g_pool_bytes += total;
                ok = 1;
                break;
            }
        }
    }
    LeaveCriticalSection(&g_cs);
    return ok;
}

// Create a unique DELETE_ON_CLOSE temp file so the backing storage is reclaimed
// when the handle closes (process exit or unmap), never surfacing on disk.
static HANDLE make_temp_file(void) {
    wchar_t path[MAX_PATH];
    static LONG64 ctr = 0;
    LONG64 n = InterlockedIncrement64(&ctr);
    _snwprintf_s(path, MAX_PATH, _TRUNCATE, L"%s\\casela-%lu-%lld.tmp",
                 g_dir, GetCurrentProcessId(), n);
    return CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                       FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
}

static void* big_alloc(size_t size, int want_zero) {
    size_t total = PAGE + ((size + PAGE - 1) & ~(PAGE - 1));

    void* base = pool_take(total);
    if (base) {
        hdr_t* h = (hdr_t*)base;   // total/hmap/hfile survive in the header
        h->magic = MAGIC; h->user = size;
        if (want_zero) memset((char*)base + PAGE, 0, size);
        InterlockedIncrement64(&g_big);
        InterlockedAdd64(&g_big_bytes, (LONG64)size);
        return (char*)base + PAGE;
    }

    HANDLE hfile = make_temp_file();
    if (hfile == INVALID_HANDLE_VALUE) { InterlockedIncrement64(&g_fallback); return NULL; }
    HANDLE hmap = CreateFileMappingW(hfile, NULL, PAGE_READWRITE,
                                     (DWORD)(total >> 32), (DWORD)(total & 0xFFFFFFFF), NULL);
    if (!hmap) { CloseHandle(hfile); InterlockedIncrement64(&g_fallback); return NULL; }
    base = MapViewOfFile(hmap, FILE_MAP_ALL_ACCESS, 0, 0, total);
    if (!base) { CloseHandle(hmap); CloseHandle(hfile); InterlockedIncrement64(&g_fallback); return NULL; }

    hdr_t* h = (hdr_t*)base;
    h->magic = MAGIC; h->user = size; h->total = total; h->hmap = hmap; h->hfile = hfile;
    InterlockedIncrement64(&g_big);
    InterlockedAdd64(&g_big_bytes, (LONG64)size);
    return (char*)base + PAGE;
}

// Is `p` one of ours? Ours is (64KB-granular base) + PAGE, so page-aligned but
// carrying our magic in the header page. Probe with VirtualQuery before touch.
static hdr_t* find(void* p) {
    if (!p || ((uintptr_t)p & (PAGE - 1))) return NULL;
    void* base = (char*)p - PAGE;
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(base, &mbi, sizeof(mbi)) != sizeof(mbi)) return NULL;
    if (mbi.Type != MEM_MAPPED || mbi.State != MEM_COMMIT) return NULL;
    hdr_t* h = (hdr_t*)base;
    return h->magic == MAGIC ? h : NULL;
}

static void big_free(hdr_t* h) {
    void* base = (void*)h;
    size_t total = h->total; HANDLE hmap = h->hmap; HANDLE hfile = h->hfile;
    h->magic = 0;
    if (pool_put(base, total, hmap, hfile)) return;   // retain, reuse later
    UnmapViewOfFile(base);
    CloseHandle(hmap);
    CloseHandle(hfile);
}

// ---- detour entry points ----
static void* d_malloc(size_t n) {
    if (n >= g_threshold) { void* p = big_alloc(n, 0); if (p) return p; }
    return real_malloc(n);
}
static void d_free(void* p) {
    if (!p) return;
    hdr_t* h = find(p);
    if (h) { big_free(h); return; }
    real_free(p);
}
static void* d_calloc(size_t c, size_t s) {
    size_t n = c * s;
    if (s && n / s != c) return NULL;
    if (n >= g_threshold) { void* p = big_alloc(n, 1); if (p) return p; }
    return real_calloc(c, s);
}
static void* d_realloc(void* p, size_t n) {
    if (!p) return d_malloc(n);
    if (n == 0) { d_free(p); return NULL; }
    hdr_t* h = find(p);
    if (h) {
        if (n <= h->total - PAGE) { h->user = n; return p; }
        void* np = d_malloc(n);
        if (!np) return NULL;
        memcpy(np, p, h->user < n ? h->user : n);
        big_free(h);
        return np;
    }
    // Foreign block growing past the threshold: move it into a mapping so it
    // stops being private commit. real_msize gives the old size to copy.
    if (n >= g_threshold) {
        size_t old = real_msize(p);
        void* np = big_alloc(n, 0);
        if (np) { memcpy(np, p, old < n ? old : n); real_free(p); return np; }
        InterlockedIncrement64(&g_fallback);
    }
    return real_realloc(p, n);
}
static size_t d_msize(void* p) {
    hdr_t* h = find(p);
    if (h) return h->total - PAGE;   // usable bytes
    return real_msize(p);
}
static void* d_aligned_malloc(size_t n, size_t a) {
    if (n >= g_threshold && a <= PAGE && a && (PAGE % a) == 0) {
        void* p = big_alloc(n, 0);
        if (p) return p;
    }
    return real_aligned_malloc(n, a);
}
static void d_aligned_free(void* p) {
    if (!p) return;
    hdr_t* h = find(p);
    if (h) { big_free(h); return; }
    real_aligned_free(p);
}
static void* d_aligned_realloc(void* p, size_t n, size_t a) {
    if (!p) return d_aligned_malloc(n, a);
    if (n == 0) { d_aligned_free(p); return NULL; }
    hdr_t* h = find(p);
    if (h) {
        if (n <= h->total - PAGE && a <= PAGE) { h->user = n; return p; }
        void* np = d_aligned_malloc(n, a);
        if (!np) return NULL;
        memcpy(np, p, h->user < n ? h->user : n);
        big_free(h);
        return np;
    }
    if (n >= g_threshold && a <= PAGE && a && (PAGE % a) == 0) {
        size_t old = real_aligned_msize(p, a, 0);
        void* np = big_alloc(n, 0);
        if (np) { memcpy(np, p, old < n ? old : n); real_aligned_free(p); return np; }
    }
    return real_aligned_realloc(p, n, a);
}
static size_t d_aligned_msize(void* p, size_t a, size_t off) {
    hdr_t* h = find(p);
    if (h) return h->total - PAGE;
    return real_aligned_msize(p, a, off);
}

// ---- install / uninstall (exported, C ABI) ----
extern "C" __declspec(dllexport)
int elastic_install_win(unsigned min_mb, unsigned pool_mb, const char* dir) {
    if (g_ready) return 0;
    g_threshold = (size_t)min_mb << 20;
    g_pool_cap  = (size_t)pool_mb << 20;

    // Backing dir: caller-provided, else %TEMP%.
    if (dir && *dir) {
        MultiByteToWideChar(CP_UTF8, 0, dir, -1, g_dir, MAX_PATH);
    } else if (!GetTempPathW(MAX_PATH, g_dir)) {
        wcscpy_s(g_dir, MAX_PATH, L".");
    }
    // Strip a trailing backslash so our _snwprintf "%s\\..." is well-formed.
    size_t dl = wcslen(g_dir);
    if (dl && g_dir[dl - 1] == L'\\') g_dir[dl - 1] = 0;

    InitializeCriticalSection(&g_cs);

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&)real_malloc,          d_malloc);
    DetourAttach(&(PVOID&)real_free,            d_free);
    DetourAttach(&(PVOID&)real_calloc,          d_calloc);
    DetourAttach(&(PVOID&)real_realloc,         d_realloc);
    DetourAttach(&(PVOID&)real_msize,           d_msize);
    DetourAttach(&(PVOID&)real_aligned_malloc,  d_aligned_malloc);
    DetourAttach(&(PVOID&)real_aligned_free,    d_aligned_free);
    DetourAttach(&(PVOID&)real_aligned_realloc, d_aligned_realloc);
    DetourAttach(&(PVOID&)real_aligned_msize,   d_aligned_msize);
    LONG rc = DetourTransactionCommit();
    if (rc == NO_ERROR) g_ready = 1;
    return (int)rc;   // 0 == NO_ERROR
}

extern "C" __declspec(dllexport)
void elastic_stats_win(long long* big, long long* big_bytes,
                       long long* pool_hits, long long* fallbacks) {
    if (big)        *big = g_big;
    if (big_bytes)  *big_bytes = g_big_bytes;
    if (pool_hits)  *pool_hits = g_pool_hit;
    if (fallbacks)  *fallbacks = g_fallback;
}
