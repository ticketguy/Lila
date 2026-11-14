#include "model.h"
#include "transformer.h"
#include "q4k.h"
#include "attention.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

extern void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void weight_matvec(float *out, const LilaQuantWeight *w, const float *vec);
extern void lila_dispatch_matvec(float *out, const float *mat, const float *vec, int rows, int cols);

int lila_ensure_scratch(LilaModel *model)
{
    if (!model)
        return -1;
    LilaScratch *s = &model->scratch;
    if (s->initialized)
        return 0;

    int hidden = model->hidden_size;
    int max_inter = model->intermediate_size;
    int max_q = hidden;
    int max_kv = model->n_kv_heads * (model->head_dim > 0 ? model->head_dim : (hidden / (model->n_heads > 0 ? model->n_heads : 1)));
    int max_o_rows = hidden;
    int max_o_cols = hidden;

    for (int l = 0; l < model->n_layers; l++)
    {
        LilaLayer *layer = &model->layers[l];
        if (layer->gate_proj.rows > max_inter) max_inter = layer->gate_proj.rows;
        if (layer->up_proj.rows > max_inter)   max_inter = layer->up_proj.rows;
        if (layer->down_proj.cols > max_inter) max_inter = layer->down_proj.cols;
        if (layer->q_proj.rows > max_q)        max_q = layer->q_proj.rows;
        if (layer->k_proj.rows > max_kv)       max_kv = layer->k_proj.rows;
        if (layer->v_proj.rows > max_kv)       max_kv = layer->v_proj.rows;
        if (layer->o_proj.rows > max_o_rows)   max_o_rows = layer->o_proj.rows;
        if (layer->o_proj.cols > max_o_cols)   max_o_cols = layer->o_proj.cols;
    }

    int seq_cap = model->max_seq_len > 0 ? model->max_seq_len : 4096;
    int vocab = model->vocab_size > 0 ? model->vocab_size : 1;

    s->max_inter = max_inter;
    s->max_q_size = max_q;
    s->max_kv_size = max_kv;
    s->max_o_rows = max_o_rows;
    s->max_o_cols = max_o_cols;

    s->hidden_state = (float *)calloc((size_t)hidden, sizeof(float));
    s->residual     = (float *)calloc((size_t)hidden, sizeof(float));
    s->normed       = (float *)calloc((size_t)hidden, sizeof(float));
    s->attn_out     = (float *)calloc((size_t)hidden, sizeof(float));
    s->mlp_out      = (float *)calloc((size_t)hidden, sizeof(float));
    s->gate         = (float *)calloc((size_t)max_inter, sizeof(float));
    s->up           = (float *)calloc((size_t)max_inter, sizeof(float));
    s->q            = (float *)calloc((size_t)max_q, sizeof(float));
    s->k            = (float *)calloc((size_t)max_kv, sizeof(float));
    s->v            = (float *)calloc((size_t)max_kv, sizeof(float));
    s->attn_buf     = (float *)calloc((size_t)max_o_cols, sizeof(float));
    s->proj_out     = (float *)calloc((size_t)max_o_rows, sizeof(float));
    s->scores       = (float *)calloc((size_t)seq_cap, sizeof(float));
    s->logits       = (float *)calloc((size_t)vocab, sizeof(float));
    s->row_buf      = (float *)calloc((size_t)hidden, sizeof(float));

    if (!s->hidden_state || !s->residual || !s->normed || !s->attn_out ||
        !s->mlp_out || !s->gate || !s->up || !s->q || !s->k || !s->v ||
        !s->attn_buf || !s->proj_out || !s->scores || !s->logits || !s->row_buf)
    {
        lila_free_scratch(model);
        return -1;
    }

    s->initialized = 1;
    return 0;
}

void lila_free_scratch(LilaModel *model)
{
    if (!model)
        return;
    LilaScratch *s = &model->scratch;
    free(s->hidden_state);
    free(s->residual);
    free(s->normed);
    free(s->attn_out);
    free(s->mlp_out);
    free(s->gate);
    free(s->up);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->attn_buf);
    free(s->proj_out);
    free(s->scores);
    free(s->logits);
    free(s->row_buf);
    free(s->qx_buf);
    free(s->rope_cos);
    free(s->rope_sin);
    memset(s, 0, sizeof(*s));
}

static inline float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}

static void lila_mlp(float *output, const float *input, LilaLayer *layer, LilaScratch *scratch)
{
    int gate_rows = layer->gate_proj.rows;
    int up_rows = layer->up_proj.rows;
    int down_cols = layer->down_proj.cols;
    int active = gate_rows < up_rows ? gate_rows : up_rows;
    if (down_cols > 0 && active > down_cols)
        active = down_cols;

    int own_alloc = 0;
    float *gate, *up;
    if (scratch && scratch->initialized)
    {
        gate = scratch->gate;
        up = scratch->up;
    }
    else
    {
        int sc = layer->intermediate_size;
        if (sc < gate_rows) sc = gate_rows;
        if (sc < up_rows)   sc = up_rows;
        if (sc < down_cols) sc = down_cols;
        own_alloc = 1;
        gate = (float *)calloc((size_t)sc, sizeof(float));
        up   = (float *)calloc((size_t)sc, sizeof(float));
        if (!gate || !up)
        {
            free(gate); free(up);
            return;
        }
    }

    weight_matvec(gate, &layer->gate_proj, input);
    weight_matvec(up, &layer->up_proj, input);

    for (int i = 0; i < active; i++)
    {
        gate[i] = silu_f(gate[i]) * up[i];
    }

    weight_matvec(output, &layer->down_proj, gate);

    if (own_alloc)
    {
        free(gate);
        free(up);
    }
}

static void lila_memory_fabric(float *output, const float *input, LilaMemoryFabric *fabric,
                               int in_features, int out_features)
{
    for (int ns = 0; ns < LILA_N_NAMESPACES; ns++)
    {
        LilaLoRA *adapter = &fabric->adapters[ns];
        if (adapter->gate < 0.01f || adapter->A == NULL || adapter->rank <= 0)
            continue;

        int r = adapter->rank;
        float mid_buf[128];
        float *mid = (r <= 128) ? mid_buf : (float *)calloc((size_t)r, sizeof(float));

        for (int j = 0; j < r; j++)
        {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++)
                sum += input[i] * adapter->A[i * r + j];
            mid[j] = sum;
        }

        float scale = adapter->gate * (32.0f / (float)r);
        for (int i = 0; i < out_features; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < r; j++)
                sum += mid[j] * adapter->B[j * out_features + i];
            output[i] += sum * scale;
        }

        if (mid != mid_buf)
            free(mid);
    }
}

static void lila_transformer_block_ex(
    float *hidden_state,
    LilaLayer *layer,
    LilaKVCache *cache,
    LilaScratch *scratch,
    float rope_theta,
    float rms_eps,
    int layer_idx,
    int position)
{
    int hidden = layer->hidden_size;
    if (rms_eps <= 0.0f)
        rms_eps = 1e-6f;

    int own_alloc = 0;
    float *residual, *normed, *attn_out, *mlp_out;
    if (scratch && scratch->initialized)
    {
        residual = scratch->residual;
        normed   = scratch->normed;
        attn_out = scratch->attn_out;
        mlp_out  = scratch->mlp_out;
    }
    else
    {
        own_alloc = 1;
        residual = (float *)malloc((size_t)hidden * sizeof(float));
        normed   = (float *)malloc((size_t)hidden * sizeof(float));
        attn_out = (float *)malloc((size_t)hidden * sizeof(float));
        mlp_out  = (float *)malloc((size_t)hidden * sizeof(float));
        if (!residual || !normed || !attn_out || !mlp_out)
        {
            free(residual); free(normed); free(attn_out); free(mlp_out);
            return;
        }
    }

    memcpy(residual, hidden_state, (size_t)hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->input_layernorm, hidden, rms_eps);

    lila_attention_ex(attn_out, normed, layer, cache, scratch, rope_theta, layer_idx, position);

    lila_memory_fabric(attn_out, normed, &layer->fabric, hidden, hidden);

    for (int i = 0; i < hidden; i++)
        hidden_state[i] = residual[i] + attn_out[i];

    memcpy(residual, hidden_state, (size_t)hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->post_attention_layernorm, hidden, rms_eps);

    lila_mlp(mlp_out, normed, layer, scratch);

    for (int i = 0; i < hidden; i++)
        hidden_state[i] = residual[i] + mlp_out[i];

    if (own_alloc)
    {
        free(residual);
        free(normed);
        free(attn_out);
        free(mlp_out);
    }
}

void lila_transformer_block(
    float *hidden_state,
    LilaLayer *layer,
    LilaKVCache *cache,
    int layer_idx,
    int position)
{
    lila_transformer_block_ex(hidden_state, layer, cache, NULL, 10000.0f, 1e-6f, layer_idx, position);
}

int lila_forward(LilaModel *model, int token, int position)
{
    int hidden = model->hidden_size;

    if (token < 0 || token >= model->vocab_size)
    {
        fprintf(stderr, "ASI: Token %d out of vocab range (%d)\n", token, model->vocab_size);
        return 1;
    }

    if (lila_ensure_scratch(model) != 0)
        return 0;

    float *hidden_state = model->scratch.hidden_state;
    float *normed       = model->scratch.normed;
    float *logits       = model->scratch.logits;

    if (model->embed_quant_type == 5 && model->embed_data != NULL)
    {
        const uint8_t *row_ptr = model->embed_data + (size_t)token * (size_t)model->embed_bytes_per_row;
        dequant_q6k_row(hidden_state, row_ptr, hidden);
    }
    else if (model->token_embedding != NULL)
    {
        memcpy(hidden_state, model->token_embedding + (size_t)token * (size_t)hidden,
               (size_t)hidden * sizeof(float));
    }
    else
    {
        memset(hidden_state, 0, (size_t)hidden * sizeof(float));
    }

    /* Model-agnostic embedding scaling (1.0f for Llama/Qwen/Mistral, sqrt(hidden) only for Gemma) */
    if (model->embed_scale > 0.0f && model->embed_scale != 1.0f)
    {
        float es = model->embed_scale;
        for (int i = 0; i < hidden; i++)
            hidden_state[i] *= es;
    }

    float rope_theta = model->rope_theta > 0.0f ? model->rope_theta : 10000.0f;
    float rms_eps    = model->rms_norm_eps > 0.0f ? model->rms_norm_eps : 1e-6f;

    for (int l = 0; l < model->n_layers; l++)
    {
        lila_transformer_block_ex(hidden_state, &model->layers[l],
                                  &model->kv_cache, &model->scratch,
                                  rope_theta, rms_eps, l, position);
    }

    lila_rmsnorm_avx2(normed, hidden_state, model->final_norm, hidden, rms_eps);

    if (model->lm_head != NULL)
    {
        lila_dispatch_matvec(logits, model->lm_head, normed, model->vocab_size, hidden);
    }
    else if (model->embed_data != NULL && model->embed_quant_type == 5)
    {
        q6k_matvec(logits, model->embed_data, normed, model->vocab_size, hidden);
    }
    else
    {
        memset(logits, 0, (size_t)model->vocab_size * sizeof(float));
    }

    int next_token = 0;
    float max_val = logits[0];
    for (int i = 1; i < model->vocab_size; i++)
    {
        if (logits[i] > max_val)
        {
            max_val = logits[i];
            next_token = i;
        }
    }

    return next_token;
}
