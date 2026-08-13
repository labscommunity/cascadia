/* elastic_preload.c — the ramlab elastic-allocator posture as an LD_PRELOAD
 * shim, so it applies to UNMODIFIED OpenVINO (or any C/C++/Rust process).
 *
 * Mechanism (D-015, generalized process-wide): every allocation >= threshold
 * is served from a MAP_SHARED mmap of an unlinked O_TMPFILE. Written pages are
 * file-dirty, not anonymous — the kernel can write them back and reclaim under
 * pressure, so they stop being OOM/commit exposure. Small allocations pass
 * through to the real allocator untouched.
 *
 * This captures what #[global_allocator] cannot: OpenVINO/oneDNN are C++, and
 * their compiled-graph weight copies, KV state and scratch all arrive through
 * malloc/new/posix_memalign — which an LD_PRELOAD interposer owns.
 *
 * Big allocations carry a header PAGE (returned pointer = base + 4096, so it
 * is always page-aligned and any alignment <= 4096 is satisfied). free()
 * identifies our pointers by page alignment + mincore() + magic — no global
 * table, no locks on the hot path.
 *
 * Freed big mappings are RETAINED in a pool and reused without munmap or
 * zeroing (D-015 evictable retention): repeated transients cost no
 * mmap/ftruncate/fault churn after warmup, yet retained pages stay file-backed
 * and pager-reclaimable, so the retention is harmless under pressure. Without
 * the pool, per-inference scratch pays an mmap round trip every step — the
 * eager-return anti-pattern exp 097 measured (v1 of this shim: -35% decode).
 *
 * Env:  ELASTIC_MIN_MB   threshold in MB (default 1)
 *       ELASTIC_DIR      backing dir for the tmpfiles (default $TMPDIR or /tmp)
 *       ELASTIC_POOL_MB  max retained-mapping bytes (default 8192; 0 = no pool)
 *       ELASTIC_LOG=1    print counters at exit
 *
 * Prototype for ramlab exp 198. Linux-only (O_TMPFILE, mincore).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define PAGE 4096UL
#define MAGIC 0xE1A571CA110CULL

typedef struct {
    uint64_t magic;
    size_t   user_size;
    size_t   total;      /* header page + rounded user size */
    int      fd;
} hdr_t;

static void *(*real_malloc)(size_t);
static void  (*real_free)(void *);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static void *(*real_aligned_alloc)(size_t, size_t);
static int   (*real_posix_memalign)(void **, size_t, size_t);
static void *(*real_memalign)(size_t, size_t);

static size_t g_threshold = 1UL << 20;
static const char *g_dir = "/tmp";
static int g_log = 0;

static _Atomic uint64_t n_big, n_big_bytes, n_free_big, n_fallback;
static _Atomic uint64_t n_pool_hit, n_pool_put;

/* ---- retention pool: freed mappings kept mapped for zero-cost reuse.
 * Fixed-size table; first-fit with a <=2x waste bound. All entries stay
 * file-backed, so the kernel can still reclaim them under pressure. ---- */
#define POOL_SLOTS 256
typedef struct { void *base; size_t total; int fd; } pool_ent;
static pool_ent g_pool[POOL_SLOTS];
static size_t g_pool_bytes;
static size_t g_pool_cap = 8UL << 30;   /* replaced at init */
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;

/* ---- bootstrap arena: dlsym() itself allocates (calloc) before the real
 * symbols are resolved; serve those few early allocations from a static
 * bump arena that is never freed. ---- */
static char boot_arena[1 << 20];
static _Atomic size_t boot_off;
static int in_boot(void *p) {
    return (char *)p >= boot_arena && (char *)p < boot_arena + sizeof(boot_arena);
}
static void *boot_alloc(size_t sz) {
    size_t o = atomic_fetch_add(&boot_off, (sz + 15) & ~15UL);
    if (o + sz > sizeof(boot_arena)) abort();
    return boot_arena + o;
}

static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static void do_init(void) {
    real_malloc         = dlsym(RTLD_NEXT, "malloc");
    real_free           = dlsym(RTLD_NEXT, "free");
    real_calloc         = dlsym(RTLD_NEXT, "calloc");
    real_realloc        = dlsym(RTLD_NEXT, "realloc");
    real_aligned_alloc  = dlsym(RTLD_NEXT, "aligned_alloc");
    real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    real_memalign       = dlsym(RTLD_NEXT, "memalign");
    const char *v;
    if ((v = getenv("ELASTIC_MIN_MB")) && atol(v) > 0)
        g_threshold = (size_t)atol(v) << 20;
    if ((v = getenv("ELASTIC_DIR")) && *v) g_dir = v;
    else if ((v = getenv("TMPDIR")) && *v) g_dir = v;
    g_pool_cap = 8UL << 30;
    if ((v = getenv("ELASTIC_POOL_MB")) && atol(v) >= 0)
        g_pool_cap = (size_t)atol(v) << 20;
    g_log = (v = getenv("ELASTIC_LOG")) && *v == '1';
}
static inline void ensure_init(void) { pthread_once(&init_once, do_init); }

/* ---- big path ---- */
static void *pool_take(size_t total_needed) {
    if (!g_pool_cap) return NULL;
    pthread_mutex_lock(&g_pool_mu);
    int best = -1;
    for (int i = 0; i < POOL_SLOTS; i++) {
        if (!g_pool[i].base) continue;
        if (g_pool[i].total >= total_needed && g_pool[i].total <= 2 * total_needed
            && (best < 0 || g_pool[i].total < g_pool[best].total))
            best = i;
    }
    void *base = NULL;
    if (best >= 0) {
        base = g_pool[best].base;
        g_pool_bytes -= g_pool[best].total;
        g_pool[best].base = NULL;
        atomic_fetch_add(&n_pool_hit, 1);
    }
    pthread_mutex_unlock(&g_pool_mu);
    return base;
}

static int pool_put(void *base, size_t total, int fd) {
    if (!g_pool_cap) return 0;
    pthread_mutex_lock(&g_pool_mu);
    if (g_pool_bytes + total <= g_pool_cap) {
        for (int i = 0; i < POOL_SLOTS; i++) {
            if (!g_pool[i].base) {
                g_pool[i] = (pool_ent){base, total, fd};
                g_pool_bytes += total;
                atomic_fetch_add(&n_pool_put, 1);
                pthread_mutex_unlock(&g_pool_mu);
                return 1;
            }
        }
    }
    pthread_mutex_unlock(&g_pool_mu);
    return 0;
}

static void *big_alloc2(size_t size, int want_zero) {
    size_t total = PAGE + ((size + PAGE - 1) & ~(PAGE - 1));
    void *base = pool_take(total);
    if (base) {                          /* reuse WITHOUT zeroing (D-015) */
        hdr_t *h = (hdr_t *)base;
        h->magic = MAGIC;                /* total + fd survive in the header */
        h->user_size = size;
        if (want_zero) memset((char *)base + PAGE, 0, size);
        atomic_fetch_add(&n_big, 1);
        atomic_fetch_add(&n_big_bytes, size);
        return (char *)base + PAGE;
    }
    int fd = open(g_dir, O_TMPFILE | O_RDWR | O_EXCL, 0600);
    if (fd < 0) { atomic_fetch_add(&n_fallback, 1); return NULL; }
    if (ftruncate(fd, (off_t)total) != 0) { close(fd); return NULL; }
    base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { close(fd); atomic_fetch_add(&n_fallback, 1); return NULL; }
    hdr_t *h = (hdr_t *)base;
    h->magic = MAGIC; h->user_size = size; h->total = total; h->fd = fd;
    atomic_fetch_add(&n_big, 1);
    atomic_fetch_add(&n_big_bytes, size);
    return (char *)base + PAGE;
}

/* Is this pointer one of ours? Only possible if page-aligned; verify the
 * header page is mapped (mincore) before dereferencing. */
static hdr_t *big_hdr(void *p) {
    if (((uintptr_t)p & (PAGE - 1)) != 0) return NULL;
    char *hp = (char *)p - PAGE;
    unsigned char vec;
    if (mincore(hp, 1, &vec) != 0) return NULL;   /* unmapped -> not ours */
    hdr_t *h = (hdr_t *)hp;
    return h->magic == MAGIC ? h : NULL;
}

static void *big_alloc(size_t size) { return big_alloc2(size, 0); }

static void big_free(hdr_t *h) {
    int fd = h->fd;
    size_t total = h->total;
    h->magic = 0;                    /* total+fd stay for pooled reuse */
    atomic_fetch_add(&n_free_big, 1);
    if (pool_put(h, total, fd)) return;
    munmap(h, total);
    close(fd);
}

/* ---- interposed API ---- */
void *malloc(size_t size) {
    ensure_init();
    if (!real_malloc) return boot_alloc(size);
    if (size >= g_threshold) {
        void *p = big_alloc(size);
        if (p) return p;
    }
    return real_malloc(size);
}

void free(void *p) {
    if (!p || in_boot(p)) return;
    ensure_init();
    hdr_t *h = big_hdr(p);
    if (h) { big_free(h); return; }
    if (real_free) real_free(p);
}

void *calloc(size_t n, size_t sz) {
    ensure_init();
    if (!real_calloc) { void *p = boot_alloc(n * sz); memset(p, 0, n * sz); return p; }
    size_t bytes = n * sz;
    if (sz != 0 && bytes / sz != n) { errno = ENOMEM; return NULL; }
    if (bytes >= g_threshold) {
        void *p = big_alloc2(bytes, 1);  /* pooled reuse must be re-zeroed */
        if (p) return p;
    }
    return real_calloc(n, sz);
}

void *realloc(void *p, size_t size) {
    ensure_init();
    if (!p) return malloc(size);
    if (size == 0) { free(p); return NULL; }
    hdr_t *h = big_hdr(p);
    if (h) {
        if (size <= h->total - PAGE) { h->user_size = size; return p; }
        void *np = malloc(size);
        if (!np) return NULL;
        memcpy(np, p, h->user_size < size ? h->user_size : size);
        big_free(h);
        return np;
    }
    if (in_boot(p)) {                      /* size unknown; copy generously */
        void *np = malloc(size);
        if (np) memcpy(np, p, size);
        return np;
    }
    if (size >= g_threshold) {
        /* foreign small->big promotion: we don't know the old size, so let the
         * real allocator grow it; it stays anonymous. Counted for honesty. */
        atomic_fetch_add(&n_fallback, 1);
    }
    return real_realloc(p, size);
}

void *aligned_alloc(size_t align, size_t size) {
    ensure_init();
    if (size >= g_threshold && align <= PAGE && (PAGE % (align ? align : 1)) == 0) {
        void *p = big_alloc(size);
        if (p) return p;
    }
    return real_aligned_alloc ? real_aligned_alloc(align, size) : NULL;
}

int posix_memalign(void **out, size_t align, size_t size) {
    ensure_init();
    if (size >= g_threshold && align <= PAGE) {
        void *p = big_alloc(size);
        if (p) { *out = p; return 0; }
    }
    return real_posix_memalign ? real_posix_memalign(out, align, size) : ENOMEM;
}

void *memalign(size_t align, size_t size) {
    ensure_init();
    if (size >= g_threshold && align <= PAGE) {
        void *p = big_alloc(size);
        if (p) return p;
    }
    return real_memalign ? real_memalign(align, size) : NULL;
}

void *valloc(size_t size) { return memalign(PAGE, size); }

__attribute__((destructor)) static void report(void) {
    if (!g_log) return;
    fprintf(stderr,
            "[elastic] big_allocs=%llu (%.1f MB total) big_frees=%llu pool_hits=%llu pool_puts=%llu fallbacks=%llu threshold=%zuMB dir=%s\n",
            (unsigned long long)n_big,
            (double)n_big_bytes / 1048576.0,
            (unsigned long long)n_free_big,
            (unsigned long long)n_pool_hit,
            (unsigned long long)n_pool_put,
            (unsigned long long)n_fallback,
            g_threshold >> 20, g_dir);
}
