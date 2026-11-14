#ifndef LILA_ASI_POOL_H
#define LILA_ASI_POOL_H

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define asi_cpu_pause() _mm_pause()
#else
#define asi_cpu_pause() ((void)0)
#endif

typedef void (*AsiTaskFn)(int tid, int n_threads, void *ctx);

typedef struct {
    int n_threads;
    pthread_t *threads;
    pthread_mutex_t mu;
    pthread_cond_t cv_work;
    AsiTaskFn fn;
    void *ctx;
    atomic_int gen;
    atomic_int done_count;
    atomic_int active;
    atomic_int shutdown;
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
        int cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire);
        while (cur_gen == last_gen)
        {
            if (atomic_load_explicit(&p->shutdown, memory_order_acquire))
            {
                free(wa);
                return NULL;
            }
            if (atomic_load_explicit(&p->active, memory_order_acquire))
            {
                asi_cpu_pause();
            }
            else
            {
                pthread_mutex_lock(&p->mu);
                while ((cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire)) == last_gen &&
                       !atomic_load_explicit(&p->active, memory_order_acquire) &&
                       !atomic_load_explicit(&p->shutdown, memory_order_acquire))
                {
                    pthread_cond_wait(&p->cv_work, &p->mu);
                }
                pthread_mutex_unlock(&p->mu);
            }
            cur_gen = atomic_load_explicit(&p->gen, memory_order_acquire);
        }
        last_gen = cur_gen;

        p->fn(tid, p->n_threads, p->ctx);

        atomic_fetch_add_explicit(&p->done_count, 1, memory_order_acq_rel);
    }
}

static inline AsiThreadPool *asi_pool_create(int n_threads)
{
    if (n_threads < 1) n_threads = 1;
    AsiThreadPool *p = (AsiThreadPool *)calloc(1, sizeof(AsiThreadPool));
    if (!p) return NULL;
    p->n_threads = n_threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    atomic_init(&p->gen, 0);
    atomic_init(&p->done_count, 0);
    atomic_init(&p->active, 0);
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
    pthread_mutex_lock(&p->mu);
    atomic_store_explicit(&p->active, 1, memory_order_release);
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
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
    if (!was_active)
    {
        pthread_mutex_lock(&p->mu);
        atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
        pthread_cond_broadcast(&p->cv_work);
        pthread_mutex_unlock(&p->mu);
    }
    else
    {
        atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    }

    fn(0, p->n_threads, ctx);

    while (atomic_load_explicit(&p->done_count, memory_order_acquire) < p->n_threads - 1)
    {
        asi_cpu_pause();
    }
}

#endif /* LILA_ASI_POOL_H */
