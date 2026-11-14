#ifndef LILA_ASI_ABI_H
#define LILA_ASI_ABI_H

#include "asi_v2.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LilaAsiModel LilaAsiModel;
typedef struct LilaAsiSession LilaAsiSession;

/* Model lifecycle */
LilaAsiModel *lila_asi_open(const char *asi_path, char *err_buf, size_t err_cap);
void lila_asi_close(LilaAsiModel *model);
int lila_asi_model_info(const LilaAsiModel *model, AsiV2ModelConfig *out_cfg);
int lila_asi_get_section(const LilaAsiModel *model,
                         uint32_t section_type,
                         const void **out_ptr,
                         size_t *out_size);

/* Session lifecycle */
LilaAsiSession *lila_asi_session_create(LilaAsiModel *model,
                                         uint32_t max_ctx,
                                         uint32_t n_threads,
                                         char *err_buf,
                                         size_t err_cap);
void lila_asi_session_free(LilaAsiSession *session);
void lila_asi_session_reset(LilaAsiSession *session);
int lila_asi_session_save_prefix(LilaAsiSession *session, uint32_t n_tokens);
int lila_asi_session_restore_prefix(LilaAsiSession *session, uint32_t *out_n_tokens);

/* Inference */
int lila_asi_eval_token(LilaAsiSession *session,
                        int32_t token_id,
                        uint32_t pos,
                        float *out_logits);
int lila_asi_eval_batch(LilaAsiSession *session,
                        const int32_t *token_ids,
                        uint32_t n_tokens,
                        uint32_t pos_start,
                        float *out_last_logits);

/* Sampling (temperature <= 0.0f selects greedy argmax) */
int32_t lila_asi_sample(const float *logits,
                        uint32_t vocab_size,
                        float temperature,
                        uint32_t top_k,
                        float top_p,
                        uint64_t *rng_state);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_ABI_H */
