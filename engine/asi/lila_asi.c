#include "lila_asi.h"
#include "asi_arch.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct LilaAsiModel {
    AsiV2Cartridge cart;
    const AsiArchHandler *handler;
};

struct LilaAsiSession {
    LilaAsiModel *model;
    AsiArchSession *arch_session;
    uint32_t n_threads;
};

LilaAsiModel *lila_asi_open(const char *asi_path, char *err_buf, size_t err_cap)
{
    LilaAsiModel *m = (LilaAsiModel *)calloc(1, sizeof(LilaAsiModel));
    if (!m)
    {
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "out of memory");
        return NULL;
    }
    if (asi_v2_open(asi_path, &m->cart, err_buf, err_cap) != 0)
    {
        free(m);
        return NULL;
    }
    m->handler = asi_arch_find(m->cart.header.arch_id);
    if (!m->handler || !m->handler->init_session)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "unsupported architecture '%s'",
                     asi_v2_arch_name(m->cart.header.arch_id));
        }
        asi_v2_close(&m->cart);
        free(m);
        return NULL;
    }
    return m;
}

void lila_asi_close(LilaAsiModel *model)
{
    if (!model) return;
    asi_v2_close(&model->cart);
    free(model);
}

int lila_asi_model_info(const LilaAsiModel *model, AsiV2ModelConfig *out_cfg)
{
    if (!model || !out_cfg) return -1;
    *out_cfg = model->cart.config;
    return 0;
}

int lila_asi_model_header(const LilaAsiModel *model, AsiV2Header *out_hdr)
{
    if (!model || !out_hdr) return -1;
    *out_hdr = model->cart.header;
    return 0;
}

int lila_asi_get_section(const LilaAsiModel *model,
                         uint32_t section_type,
                         const void **out_ptr,
                         size_t *out_size)
{
    if (!model || !out_ptr || !out_size) return -1;
    const AsiV2SectionEntry *sec = asi_v2_find_section(&model->cart, section_type);
    if (!sec) return -1;
    *out_ptr = model->cart.base + sec->offset;
    *out_size = (size_t)sec->size;
    return 0;
}

LilaAsiSession *lila_asi_session_create(LilaAsiModel *model,
                                         uint32_t max_ctx,
                                         uint32_t n_threads,
                                         char *err_buf,
                                         size_t err_cap)
{
    if (!model || !model->handler) return NULL;
    LilaAsiSession *s = (LilaAsiSession *)calloc(1, sizeof(LilaAsiSession));
    if (!s) return NULL;
    s->model = model;
    if (n_threads == 0)
    {
        int procs = omp_get_num_procs();
        n_threads = (procs > 2) ? (uint32_t)(procs / 2) : (uint32_t)procs;
        if (n_threads == 0) n_threads = 1;
    }
    s->n_threads = n_threads;
    omp_set_num_threads((int)n_threads);

    if (model->handler->init_session(&model->cart, max_ctx, n_threads,
                                     &s->arch_session, err_buf, err_cap) != 0)
    {
        free(s);
        return NULL;
    }
    return s;
}

void lila_asi_session_free(LilaAsiSession *session)
{
    if (!session) return;
    if (session->model && session->model->handler && session->model->handler->free_session)
    {
        session->model->handler->free_session(session->arch_session);
    }
    free(session);
}

void lila_asi_session_reset(LilaAsiSession *session)
{
    if (!session || !session->model || !session->model->handler) return;
    if (session->model->handler->reset_state)
    {
        session->model->handler->reset_state(session->arch_session);
    }
}

int lila_asi_session_save_prefix(LilaAsiSession *session, uint32_t n_tokens)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->save_prefix) return -1;
    return session->model->handler->save_prefix(session->arch_session, n_tokens);
}

int lila_asi_session_restore_prefix(LilaAsiSession *session, uint32_t *out_n_tokens)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->restore_prefix) return -1;
    return session->model->handler->restore_prefix(session->arch_session, out_n_tokens);
}

int lila_asi_session_save_prefix_file(const LilaAsiSession *session,
                                      const uint8_t key_hash[32],
                                      const char *filepath)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->save_prefix_file) return -1;
    return session->model->handler->save_prefix_file(session->arch_session, key_hash, filepath);
}

int lila_asi_session_load_prefix_file(LilaAsiSession *session,
                                      const uint8_t expected_key_hash[32],
                                      const char *filepath,
                                      uint32_t *out_n_tokens)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->load_prefix_file) return -1;
    return session->model->handler->load_prefix_file(session->arch_session,
                                                     expected_key_hash,
                                                     filepath,
                                                     out_n_tokens);
}

int lila_asi_eval_token(LilaAsiSession *session,
                        int32_t token_id,
                        uint32_t pos,
                        float *out_logits)
{
    if (!session || !session->model || !session->model->handler) return -1;
    omp_set_num_threads((int)session->n_threads);
    if (session->model->handler->forward_speculative)
    {
        return session->model->handler->forward_speculative(
            session->arch_session, token_id, pos, out_logits);
    }
    return session->model->handler->forward_token(session->arch_session, token_id, pos, out_logits);
}

int lila_asi_eval_batch(LilaAsiSession *session,
                        const int32_t *token_ids,
                        uint32_t n_tokens,
                        uint32_t pos_start,
                        float *out_last_logits)
{
    if (!session || !session->model || !session->model->handler) return -1;
    omp_set_num_threads((int)session->n_threads);
    return session->model->handler->forward_batch(
        session->arch_session, token_ids, n_tokens, pos_start, out_last_logits);
}

typedef struct {
    int32_t id;
    float logit;
} CandLogit;

static int cmp_cand_desc(const void *a, const void *b)
{
    float la = ((const CandLogit *)a)->logit;
    float lb = ((const CandLogit *)b)->logit;
    return (lb > la) - (lb < la);
}

static float rng_uniform(uint64_t *state)
{
    uint64_t x = (*state) ? *state : 0x9E3779B97F4A7C15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1ULL << 24);
}

int32_t lila_asi_sample(const float *logits,
                        uint32_t vocab_size,
                        float temperature,
                        uint32_t top_k,
                        float top_p,
                        uint64_t *rng_state)
{
    if (!logits || vocab_size == 0) return -1;
    if (temperature <= 0.0f)
    {
        int32_t best = 0;
        float best_v = logits[0];
        for (uint32_t i = 1; i < vocab_size; ++i)
        {
            if (logits[i] > best_v)
            {
                best_v = logits[i];
                best = (int32_t)i;
            }
        }
        return best;
    }

    uint32_t k = (top_k > 0 && top_k < vocab_size) ? top_k : 64;
    CandLogit *cands = (CandLogit *)malloc((size_t)k * sizeof(CandLogit));
    if (!cands) return 0;

    uint32_t count = 0;
    float min_v = -1e30f;
    uint32_t min_idx = 0;
    for (uint32_t i = 0; i < vocab_size; ++i)
    {
        float v = logits[i];
        if (count < k)
        {
            cands[count].id = (int32_t)i;
            cands[count].logit = v;
            if (count == 0 || v < min_v)
            {
                min_v = v;
                min_idx = count;
            }
            count++;
        }
        else if (v > min_v)
        {
            cands[min_idx].id = (int32_t)i;
            cands[min_idx].logit = v;
            min_v = cands[0].logit;
            min_idx = 0;
            for (uint32_t j = 1; j < k; ++j)
            {
                if (cands[j].logit < min_v)
                {
                    min_v = cands[j].logit;
                    min_idx = j;
                }
            }
        }
    }

    qsort(cands, count, sizeof(CandLogit), cmp_cand_desc);

    float max_l = cands[0].logit;
    float sum_p = 0.0f;
    for (uint32_t i = 0; i < count; ++i)
    {
        cands[i].logit = expf((cands[i].logit - max_l) / temperature);
        sum_p += cands[i].logit;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        cands[i].logit /= sum_p;
    }

    if (top_p > 0.0f && top_p < 1.0f)
    {
        float cum = 0.0f;
        uint32_t cut = count;
        for (uint32_t i = 0; i < count; ++i)
        {
            cum += cands[i].logit;
            if (cum >= top_p)
            {
                cut = i + 1;
                break;
            }
        }
        count = cut;
        float renorm = 0.0f;
        for (uint32_t i = 0; i < count; ++i) renorm += cands[i].logit;
        for (uint32_t i = 0; i < count; ++i) cands[i].logit /= renorm;
    }

    float r = rng_uniform(rng_state);
    float acc = 0.0f;
    int32_t chosen = cands[count - 1].id;
    for (uint32_t i = 0; i < count; ++i)
    {
        acc += cands[i].logit;
        if (r <= acc)
        {
            chosen = cands[i].id;
            break;
        }
    }
    free(cands);
    return chosen;
}
