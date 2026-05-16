#include "model.h"
#include "q4k.h"
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

    float *gate = calloc((size_t)scratch, sizeof(float));
    float *up = calloc((size_t)scratch, sizeof(float));
    if (!gate || !up)
    {
        fprintf(stderr, "ASI: MLP scratch allocation failed (%d floats)\n", scratch);
        free(gate);
        free(up);
        return;
    }

    
    weight_matvec(gate, &layer->gate_proj, input);
    
    
    weight_matvec(up, &layer->up_proj, input);
    

    for (int i = 0; i < active; i++)
    {
        gate[i] = silu_f(gate[i]) * up[i];
    }

    
    weight_matvec(output, &layer->down_proj, gate);
    

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

    float *residual = malloc(hidden * sizeof(float));
    float *normed = malloc(hidden * sizeof(float));
    float *attn_out = malloc(hidden * sizeof(float));
    float *mlp_out = malloc(hidden * sizeof(float));

    /* Pre-attention norm */
    memcpy(residual, hidden_state, hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->input_layernorm, hidden, 1e-6f);
    

    /* Attention */
    
    lila_attention(attn_out, normed, layer, cache, layer_idx, position);
    

    /* Memory Fabric */
    lila_memory_fabric(attn_out, normed, &layer->fabric, hidden, hidden);
    

    /* Residual */
    for (int i = 0; i < hidden; i++)
        hidden_state[i] = residual[i] + attn_out[i];

    /* Pre-MLP norm */
    memcpy(residual, hidden_state, hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->post_attention_layernorm, hidden, 1e-6f);
    

    /* MLP */
    lila_mlp(mlp_out, normed, layer);
    

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

    /* Bounds check token */
    if (token < 0 || token >= model->vocab_size)
    {
        fprintf(stderr, "ASI: Token %d out of vocab range (%d)\n", token, model->vocab_size);
        return 1; /* return BOS as safe fallback */
    }

    /* Token embedding — handle both FP32 and quantized (Q6_K) */
    float *hidden_state = malloc(hidden * sizeof(float));
    
    if (model->embed_quant_type == 5 && model->embed_data != NULL) {
        /* Q6_K quantized embedding — dequant one row on the fly */
        const uint8_t *row_ptr = model->embed_data + (size_t)token * model->embed_bytes_per_row;
        dequant_q6k_row(hidden_state, row_ptr, hidden);
    } else if (model->token_embedding != NULL) {
        /* FP32 embedding — direct copy */
        memcpy(hidden_state, model->token_embedding + (size_t)token * hidden,
               hidden * sizeof(float));
    } else {
        /* No embedding available */
        memset(hidden_state, 0, hidden * sizeof(float));
    }

    /* Gemma embedding scale: multiply by sqrt(hidden_size) */
    float embed_scale = sqrtf((float)hidden);
    for (int i = 0; i < hidden; i++) {
        hidden_state[i] *= embed_scale;
    }

    /* Transformer layers */
    for (int l = 0; l < model->n_layers; l++)
    {
        lila_transformer_block(hidden_state, &model->layers[l],
                               &model->kv_cache, l, position);
    }

    

    /* Final norm */
    float *normed = malloc(hidden * sizeof(float));
    
    lila_rmsnorm_avx2(normed, hidden_state, model->final_norm, hidden, 1e-6f);

    /* LM head: [vocab_size, hidden] @ normed → logits */
    float *logits = malloc(model->vocab_size * sizeof(float));
    
    if (model->lm_head != NULL) {
        /* FP32 LM head — direct matvec */
        lila_dispatch_matvec(logits, model->lm_head, normed, model->vocab_size, hidden);
    } else if (model->embed_data != NULL && model->embed_quant_type == 5) {
        /* Weight-tied Q6_K embedding — dequant each vocab row and dot with normed.
         * This is the bottleneck for Gemma (262144 rows) but it's correct. */
        int bytes_per_row = model->embed_bytes_per_row;
        float row_buf[2560]; /* stack buffer for one dequanted row (hidden <= 2560) */
        
        #pragma omp parallel for schedule(static) private(row_buf)
        for (int i = 0; i < model->vocab_size; i++) {
            const uint8_t *row_ptr = model->embed_data + (size_t)i * bytes_per_row;
            dequant_q6k_row(row_buf, row_ptr, hidden);
            float sum = 0.0f;
            for (int j = 0; j < hidden; j++) {
                sum += row_buf[j] * normed[j];
            }
            logits[i] = sum;
        }
    } else {
        /* No LM head available — return EOS */
        memset(logits, 0, model->vocab_size * sizeof(float));
    }

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

    free(hidden_state);
    free(normed);
    free(logits);

    return next_token;
}
