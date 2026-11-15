#ifndef LILA_ASI_POOL_H
#define LILA_ASI_POOL_H

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __linux__
#include <dirent.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define asi_cpu_pause() _mm_pause()
#else
#define asi_cpu_pause() ((void)0)
#endif

typedef void (*AsiTaskFn)(int tid, int n_threads, void *ctx);

typedef struct {
    int n_threads;
    int spin_limit;
    int yield_limit;
    pthread_t *threads;
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    _Alignas(64) AsiTaskFn fn;
    void *ctx;
    _Alignas(64) atomic_int gen;
    _Alignas(64) atomic_int done_count;
    _Alignas(64) atomic_int active;
    _Alignas(64) atomic_int sleeping;
    _Alignas(64) atomic_int shutdown;
} AsiThreadPool;

typedef struct {
    AsiThreadPool *pool;
    int tid;
} AsiWorkerArg;

static void *asi_pool_worker(void *arg)
{
    AsiWorkerArg *wa = (AsiWorkerArg *)arg;
    AsiThreadPool *p = wa->pool;
    int tid = wa->tid;
    int last_gen = 0;

    for (;;)
    {
        int spins = 0;
        int cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire);
        while (cur_gen == last_gen)
        {
            if (atomic_load_explicit(&p->shutdown, memory_order_acquire))
            {
                free(wa);
                return NULL;
            }
            int active = atomic_load_explicit(&p->active, memory_order_acquire);
            if (active && spins < p->spin_limit)
            {
                asi_cpu_pause();
                spins++;
            }
            else if (active && spins < p->yield_limit)
            {
                sched_yield();
                spins++;
            }
            else
            {
                pthread_mutex_lock(&p->mu);
                atomic_fetch_add_explicit(&p->sleeping, 1, memory_order_seq_cst);
                while ((cur_gen = atomic_load_explicit(&p->gen, memory_order_seq_cst)) == last_gen &&
                       !atomic_load_explicit(&p->shutdown, memory_order_acquire))
                {
                    pthread_cond_wait(&p->cv_work, &p->mu);
                }
                atomic_fetch_sub_explicit(&p->sleeping, 1, memory_order_release);
                pthread_mutex_unlock(&p->mu);
                spins = 0;
                continue;
            }
            cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire);
        }
        last_gen = cur_gen;

        p->fn(tid, p->n_threads, p->ctx);

        atomic_fetch_add_explicit(&p->done_count, 1, memory_order_acq_rel);
    }
}

static inline int asi_detect_physical_cores(void)
{
#ifdef __linux__
    DIR *dir = opendir("/sys/devices/system/cpu");
    if (dir)
    {
        char seen[512][32];
        int n_seen = 0;
        struct dirent *ent;
        while ((ent = readdir(dir)) != NULL)
        {
            if (strncmp(ent->d_name, "cpu", 3) != 0) continue;
            const char *p = ent->d_name + 3;
            if (*p < '0' || *p > '9') continue;
            int all_digits = 1;
            for (const char *q = p; *q; q++)
            {
                if (*q < '0' || *q > '9') { all_digits = 0; break; }
            }
            if (!all_digits) continue;

            char path[512];
            int pn = snprintf(path, sizeof(path),
                              "/sys/devices/system/cpu/%s/topology/thread_siblings_list",
                              ent->d_name);
            if (pn <= 0 || (size_t)pn >= sizeof(path)) continue;

            FILE *f = fopen(path, "r");
            if (!f) continue;
            char buf[32];
            if (fgets(buf, sizeof(buf), f))
            {
                size_t len = strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r' || buf[len - 1] == ' '))
                {
                    buf[--len] = '\0';
                }
                if (len > 0)
                {
                    int dup = 0;
                    for (int i = 0; i < n_seen; i++)
                    {
                        if (strcmp(seen[i], buf) == 0) { dup = 1; break; }
                    }
                    if (!dup && n_seen < 512)
                    {
                        memcpy(seen[n_seen], buf, len + 1);
                        n_seen++;
                    }
                }
            }
            fclose(f);
        }
        closedir(dir);
        if (n_seen > 0) return n_seen;
    }
    long n_onln = sysconf(_SC_NPROCESSORS_ONLN);
    return (n_onln > 1) ? (int)(n_onln / 2) : 1;
#else
    return 1;
#endif
}

static inline AsiThreadPool *asi_pool_create(int n_threads)
{
    if (n_threads < 1) n_threads = 1;
    AsiThreadPool *p = (AsiThreadPool *)aligned_alloc(64, sizeof(AsiThreadPool));
    if (!p) return NULL;
    memset(p, 0, sizeof(AsiThreadPool));
    p->n_threads = n_threads;

    int phys = asi_detect_physical_cores();
    p->spin_limit = (n_threads > phys) ? 64 : 2048;
    p->yield_limit = (n_threads > phys) ? 128 : 4096;
    const char *env_spin = getenv("LILA_ASI_SPIN_LIMIT");
    if (env_spin && env_spin[0] != '\0')
    {
        char *endp = NULL;
        long val = strtol(env_spin, &endp, 10);
        if (endp != env_spin && *endp == '\0' && val >= 0)
        {
            p->spin_limit = (val > 1000000L) ? 1000000 : (int)val;
            p->yield_limit = (p->spin_limit > 0) ? (p->spin_limit * 2) : 0;
        }
    }

    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    atomic_init(&p->gen, 0);
    atomic_init(&p->done_count, 0);
    atomic_init(&p->active, 0);
    atomic_init(&p->sleeping, 0);
    atomic_init(&p->shutdown, 0);

    if (n_threads > 1)
    {
        p->threads = (pthread_t *)calloc((size_t)(n_threads - 1), sizeof(pthread_t));
        for (int t = 1; t < n_threads; ++t)
        {
            AsiWorkerArg *wa = (AsiWorkerArg *)malloc(sizeof(AsiWorkerArg));
            wa->pool = p;
            wa->tid = t;
            pthread_create(&p->threads[t - 1], NULL, asi_pool_worker, wa);
        }
    }
    return p;
}

static inline void asi_pool_destroy(AsiThreadPool *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store_explicit(&p->shutdown, 1, memory_order_release);
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    for (int t = 1; t < p->n_threads; ++t)
    {
        pthread_join(p->threads[t - 1], NULL);
    }
    free(p->threads);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    free(p);
}

static inline void asi_pool_begin(AsiThreadPool *p)
{
    if (!p || p->n_threads <= 1) return;
    atomic_store_explicit(&p->active, 1, memory_order_release);
}

static inline void asi_pool_end(AsiThreadPool *p)
{
    if (!p || p->n_threads <= 1) return;
    atomic_store_explicit(&p->active, 0, memory_order_release);
}

static inline void asi_pool_run(AsiThreadPool *p, AsiTaskFn fn, void *ctx)
{
    if (!p || p->n_threads <= 1)
    {
        fn(0, 1, ctx);
        return;
    }
    int was_active = atomic_load_explicit(&p->active, memory_order_acquire);
    p->fn = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->done_count, 0, memory_order_release);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_seq_cst);
    if (!was_active || atomic_load_explicit(&p->sleeping, memory_order_seq_cst) > 0)
    {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv_work);
        pthread_mutex_unlock(&p->mu);
    }

    fn(0, p->n_threads, ctx);

    int spins = 0;
    while (atomic_load_explicit(&p->done_count, memory_order_acquire) < p->n_threads - 1)
    {
        if (spins < p->spin_limit)
        {
            asi_cpu_pause();
            spins++;
        }
        else
        {
            sched_yield();
        }
    }
}

#endif /* LILA_ASI_POOL_H */
