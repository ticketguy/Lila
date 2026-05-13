#ifndef LILA_ATTENTION_H
#define LILA_ATTENTION_H

#include "model.h"

void lila_init_kv_cache(LilaKVCache *cache, int n_layers, int max_seq,
                         int n_kv_heads, int head_dim);
void lila_attention(float *output, const float *input, LilaLayer *layer,
                    LilaKVCache *cache, int layer_idx, int position);

#endif
