#include "model.h"
#include <math.h>
#include <stdio.h>

/* External declarations */
extern void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void lila_attention(float *output, const float *input, LilaLayer *layer,
                           LilaKVCache *cache, int layer_idx, int position);
extern void weight_matvec(float *out, const LilaQuantWeight *w, const float *vec);
extern void lila_dispatch_matvec(float *out, const float *mat, const float *vec, int rows, int cols);

#include <stdlib.h>
#include <string.h>

/*
 * Full transformer decoder block:
 *   residual = x
 *   x = rmsnorm(x)
 *   x = attention(x) + residual
 *   residual = x
 *   x = rmsnorm(x)
 *   x = mlp(x) + residual
 */

/* SiLU activation */
static inline float silu_f(float x)
{
    return x / (1.0f + expf(-x));
}

/* MLP: gate_proj + up_proj → SiLU(gate) * up → down_proj */
static void lila_mlp(float *output, const float *input, LilaLayer *layer)
{
    int configured_inter = layer->intermediate_size;
    int gate_rows = layer->gate_proj.rows;
    int up_rows = layer->up_proj.rows;
    int down_cols = layer->down_proj.cols;
    int scratch = configured_inter;

    if (scratch < gate_rows)
        scratch = gate_rows;
    if (scratch < up_rows)
        scratch = up_rows;
    if (scratch < down_cols)
        scratch = down_cols;

    int active = gate_rows < up_rows ? gate_rows : up_rows;
    if (down_cols > 0 && active > down_cols)
        active = down_cols;

    fprintf(stderr, "[DBG] mlp: configured_inter=%d scratch=%d active=%d gate=%dx%d up=%dx%d down=%dx%d\n",
            configured_inter, scratch, active,
            layer->gate_proj.rows, layer->gate_proj.cols,
            layer->up_proj.rows, layer->up_proj.cols,
            layer->down_proj.rows, layer->down_proj.cols);
    fflush(stderr);

    float *gate = calloc((size_t)scratch, sizeof(float));
    float *up = calloc((size_t)scratch, sizeof(float));
    if (!gate || !up)
    {
        fprintf(stderr, "ASI: MLP scratch allocation failed (%d floats)\n", scratch);
        free(gate);
        free(up);
        return;
    }

    fprintf(stderr, "[DBG] mlp calling gate weight_matvec...\n");
    fflush(stderr);
    weight_matvec(gate, &layer->gate_proj, input);
    fprintf(stderr, "[DBG] mlp gate returned\n");
    fflush(stderr);
    fprintf(stderr, "[DBG] mlp calling up weight_matvec...\n");
    fflush(stderr);
    weight_matvec(up, &layer->up_proj, input);
    fprintf(stderr, "[DBG] mlp up returned\n");
    fflush(stderr);

    for (int i = 0; i < active; i++)
    {
        gate[i] = silu_f(gate[i]) * up[i];
    }

    fprintf(stderr, "[DBG] mlp calling down weight_matvec...\n");
    fflush(stderr);
    weight_matvec(output, &layer->down_proj, gate);
    fprintf(stderr, "[DBG] mlp down returned\n");
    fflush(stderr);

    free(gate);
    free(up);
}

/* Memory Fabric contribution (multi-LoRA gated adapters) */
static void lila_memory_fabric(float *output, const float *input, LilaMemoryFabric *fabric,
                               int in_features, int out_features)
{
    for (int ns = 0; ns < LILA_N_NAMESPACES; ns++)
    {
        LilaLoRA *adapter = &fabric->adapters[ns];
        if (adapter->gate < 0.01f || adapter->A == NULL)
            continue;

        int r = adapter->rank;

        float *mid = calloc(r, sizeof(float));

        for (int j = 0; j < r; j++)
        {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++)
            {
                sum += input[i] * adapter->A[i * r + j];
            }
            mid[j] = sum;
        }

        float scale = adapter->gate * (32.0f / r);
        for (int i = 0; i < out_features; i++)
        {
            float sum = 0.0f;
            for (int j = 0; j < r; j++)
            {
                sum += mid[j] * adapter->B[j * out_features + i];
            }
            output[i] += sum * scale;
        }

        free(mid);
    }
}

/* Full transformer block forward pass */
void lila_transformer_block(
    float *hidden_state,
    LilaLayer *layer,
    LilaKVCache *cache,
    int layer_idx,
    int position)
{
    int hidden = layer->hidden_size;
    fprintf(stderr, "[DBG] block layer=%d hidden=%d inter=%d head_dim=%d\n",
            layer_idx, hidden, layer->intermediate_size, layer->head_dim);
    fprintf(stderr, "[DBG] input_layernorm=%p\n", (void *)layer->input_layernorm);
    fprintf(stderr, "[DBG] calling rmsnorm...\n");
    fflush(stderr);

    float *residual = malloc(hidden * sizeof(float));
    float *normed = malloc(hidden * sizeof(float));
    float *attn_out = malloc(hidden * sizeof(float));
    float *mlp_out = malloc(hidden * sizeof(float));

    /* Pre-attention norm */
    memcpy(residual, hidden_state, hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->input_layernorm, hidden, 1e-6f);
    fprintf(stderr, "[DBG] rmsnorm returned\n");
    fflush(stderr);

    /* Attention */
    fprintf(stderr, "[DBG] calling attention...\n");
    fflush(stderr);
    lila_attention(attn_out, normed, layer, cache, layer_idx, position);
    fprintf(stderr, "[DBG] attention returned\n");
    fflush(stderr);

    /* Memory Fabric */
    lila_memory_fabric(attn_out, normed, &layer->fabric, hidden, hidden);
    fprintf(stderr, "[DBG] memory_fabric returned\n");
    fflush(stderr);

    /* Residual */
    for (int i = 0; i < hidden; i++)
        hidden_state[i] = residual[i] + attn_out[i];

    /* Pre-MLP norm */
    memcpy(residual, hidden_state, hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->post_attention_layernorm, hidden, 1e-6f);
    fprintf(stderr, "[DBG] post_attn rmsnorm returned\n");
    fflush(stderr);

    /* MLP */
    lila_mlp(mlp_out, normed, layer);
    fprintf(stderr, "[DBG] mlp returned\n");
    fflush(stderr);

    /* Residual */
    for (int i = 0; i < hidden; i++)
        hidden_state[i] = residual[i] + mlp_out[i];

    free(residual);
    free(normed);
    free(attn_out);
    free(mlp_out);
}

/* Full model forward pass — single token */
int lila_forward(LilaModel *model, int token, int position)
{
    int hidden = model->hidden_size;

    fprintf(stderr, "[DBG] lila_forward: token=%d pos=%d hidden=%d vocab=%d\n",
            token, position, hidden, model->vocab_size);
    fprintf(stderr, "[DBG] embedding ptr: %p\n", (void *)model->token_embedding);
    fprintf(stderr, "[DBG] final_norm ptr: %p\n", (void *)model->final_norm);
    fprintf(stderr, "[DBG] lm_head ptr: %p\n", (void *)model->lm_head);
    fprintf(stderr, "[DBG] layer[0] q_proj: data=%p indices=%p rows=%d cols=%d\n",
            (void *)model->layers[0].q_proj.data,
            (void *)model->layers[0].q_proj.indices,
            model->layers[0].q_proj.rows,
            model->layers[0].q_proj.cols);
    fprintf(stderr, "[DBG] layer[0] input_layernorm: %p\n",
            (void *)model->layers[0].input_layernorm);

    /* Bounds check token */
    if (token < 0 || token >= model->vocab_size)
    {
        fprintf(stderr, "ASI: Token %d out of vocab range (%d)\n", token, model->vocab_size);
        return 1; /* return BOS as safe fallback */
    }

    /* Token embedding */
    float *hidden_state = malloc(hidden * sizeof(float));
    memcpy(hidden_state, model->token_embedding + (size_t)token * hidden,
           hidden * sizeof(float));

    /* Transformer layers */
    for (int l = 0; l < model->n_layers; l++)
    {
        lila_transformer_block(hidden_state, &model->layers[l],
                               &model->kv_cache, l, position);
    }

    fprintf(stderr, "[DBG] all transformer layers returned\n");
    fflush(stderr);

    /* Final norm */
    float *normed = malloc(hidden * sizeof(float));
    fprintf(stderr, "[DBG] calling final rmsnorm...\n");
    fflush(stderr);
    lila_rmsnorm_avx2(normed, hidden_state, model->final_norm, hidden, 1e-6f);
    fprintf(stderr, "[DBG] final rmsnorm returned\n");
    fflush(stderr);

    /* LM head: [vocab_size, hidden] @ normed → logits
     * Use dispatch so AVX2 is used where available */
    float *logits = malloc(model->vocab_size * sizeof(float));
    fprintf(stderr, "[DBG] calling lm_head matvec...\n");
    fflush(stderr);
    lila_dispatch_matvec(logits, model->lm_head, normed, model->vocab_size, hidden);
    fprintf(stderr, "[DBG] lm_head matvec returned\n");
    fflush(stderr);

    /* Greedy argmax sampling */
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

    fprintf(stderr, "[DBG] sampled next_token=%d max=%f\n", next_token, max_val);
    fflush(stderr);

    free(hidden_state);
    free(normed);
    free(logits);

    return next_token;
}
