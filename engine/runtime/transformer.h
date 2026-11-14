#ifndef LILA_TRANSFORMER_H
#define LILA_TRANSFORMER_H

#include "model.h"

int lila_ensure_scratch(LilaModel *model);
void lila_free_scratch(LilaModel *model);
void lila_transformer_block(float *hidden_state, LilaLayer *layer,
                            LilaKVCache *cache, int layer_idx, int position);
int lila_forward(LilaModel *model, int token, int position);

#endif
