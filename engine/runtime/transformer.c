#include "model.h"
#include <math.h>

/* External declarations */
extern void lila_rmsnorm_avx2(float *out, const float *x, const float *weight, int size, float eps);
extern void lila_attention(float *output, const float *input, LilaLayer *layer,
                           LilaKVCache *cache, int layer_idx, int position);
extern void dequant_matvec(float *out, const LilaQuantWeight *w, const float *vec);

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

/* External kernel declarations */




/* SiLU activation (will be assembly in Phase 4) */
static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

/* MLP: gate_proj + up_proj → SiLU(gate) * up → down_proj */
static void lila_mlp(float *output, const float *input, LilaLayer *layer) {
    int hidden = layer->hidden_size;
    int inter = layer->intermediate_size;
    
    float *gate = malloc(inter * sizeof(float));
    float *up = malloc(inter * sizeof(float));
    
    /* Gate and up projections */
    dequant_matvec(gate, &layer->gate_proj, input);
    dequant_matvec(up, &layer->up_proj, input);
    
    /* SiLU(gate) * up */
    for (int i = 0; i < inter; i++) {
        gate[i] = silu_f(gate[i]) * up[i];
    }
    
    /* Down projection */
    dequant_matvec(output, &layer->down_proj, gate);
    
    free(gate);
    free(up);
}

/* Memory Fabric contribution (multi-LoRA gated adapters) */
static void lila_memory_fabric(float *output, const float *input, LilaMemoryFabric *fabric,
                                int in_features, int out_features) {
    /* For each active namespace adapter, compute gated LoRA correction */
    for (int ns = 0; ns < LILA_N_NAMESPACES; ns++) {
        LilaLoRA *adapter = &fabric->adapters[ns];
        if (adapter->gate < 0.01f || adapter->A == NULL) continue;
        
        int r = adapter->rank;
        
        /* Compute: gate * (input @ A) @ B */
        float *mid = calloc(r, sizeof(float));
        
        /* mid = input @ A  [in_features] @ [in_features, r] → [r] */
        for (int j = 0; j < r; j++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += input[i] * adapter->A[i * r + j];
            }
            mid[j] = sum;
        }
        
        /* output += gate * (mid @ B)  [r] @ [r, out_features] → [out_features] */
        float scale = adapter->gate * (32.0f / r);  /* alpha/rank */
        for (int i = 0; i < out_features; i++) {
            float sum = 0.0f;
            for (int j = 0; j < r; j++) {
                sum += mid[j] * adapter->B[j * out_features + i];
            }
            output[i] += sum * scale;
        }
        
        free(mid);
    }
}

/* Full transformer block forward pass */
void lila_transformer_block(
    float *hidden_state,    /* [hidden_size] — modified in place */
    LilaLayer *layer,
    LilaKVCache *cache,
    int layer_idx,
    int position
) {
    int hidden = layer->hidden_size;
    float *residual = malloc(hidden * sizeof(float));
    float *normed = malloc(hidden * sizeof(float));
    float *attn_out = malloc(hidden * sizeof(float));
    float *mlp_out = malloc(hidden * sizeof(float));
    
    /* ── Pre-attention norm ── */
    memcpy(residual, hidden_state, hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->input_layernorm, hidden, 1e-6f);
    
    /* ── Attention ── */
    lila_attention(attn_out, normed, layer, cache, layer_idx, position);
    
    /* ── Add Memory Fabric to attention output ── */
    lila_memory_fabric(attn_out, normed, &layer->fabric, hidden, hidden);
    
    /* ── Residual connection ── */
    for (int i = 0; i < hidden; i++) hidden_state[i] = residual[i] + attn_out[i];
    
    /* ── Pre-MLP norm ── */
    memcpy(residual, hidden_state, hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, layer->post_attention_layernorm, hidden, 1e-6f);
    
    /* ── MLP ── */
    lila_mlp(mlp_out, normed, layer);
    
    /* ── Residual connection ── */
    for (int i = 0; i < hidden; i++) hidden_state[i] = residual[i] + mlp_out[i];
    
    free(residual);
    free(normed);
    free(attn_out);
    free(mlp_out);
}

/* Full model forward pass — single token */
int lila_forward(LilaModel *model, int token, int position) {
    int hidden = model->hidden_size;
    
    /* Token embedding */
    float *hidden_state = malloc(hidden * sizeof(float));
    memcpy(hidden_state, model->token_embedding + (size_t)token * hidden,
           hidden * sizeof(float));
    
    /* Transformer layers */
    for (int l = 0; l < model->n_layers; l++) {
        lila_transformer_block(hidden_state, &model->layers[l],
                               &model->kv_cache, l, position);
    }
    
    /* Final norm */
    float *normed = malloc(hidden * sizeof(float));
    lila_rmsnorm_avx2(normed, hidden_state, model->final_norm, hidden, 1e-6f);
    
    /* LM head: project to vocab logits */
    float *logits = malloc(model->vocab_size * sizeof(float));
    
    /* matvec: logits = lm_head @ normed */
    /* lm_head is [vocab_size, hidden_size] */
    for (int i = 0; i < model->vocab_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += model->lm_head[i * hidden + j] * normed[j];
        }
        logits[i] = sum;
    }
    
    /* Sample */
    /* Greedy for now — temperature sampling in Phase 4 */
    int next_token = 0;
    float max_val = logits[0];
    for (int i = 1; i < model->vocab_size; i++) {
        if (logits[i] > max_val) { max_val = logits[i]; next_token = i; }
    }
    
    free(hidden_state);
    free(normed);
    free(logits);
    
    return next_token;
}
