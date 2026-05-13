#ifndef LILA_TOKENIZER_H
#define LILA_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct LilaTokenizer LilaTokenizer;

LilaTokenizer *lila_load_tokenizer(const char *vocab_path);
LilaTokenizer *lila_load_tokenizer_from_memory(const uint8_t *data, size_t data_size,
                                                int vocab_size, int bos_id, int eos_id, int pad_id);
const char *lila_decode_token(LilaTokenizer *tok, int token_id);
char *lila_decode_sequence(LilaTokenizer *tok, const int *tokens, int n_tokens);
int lila_encode(LilaTokenizer *tok, const char *text, int *output_ids, int max_tokens);
int lila_get_bos(LilaTokenizer *tok);
int lila_get_eos(LilaTokenizer *tok);
int lila_get_vocab_size(LilaTokenizer *tok);
void lila_free_tokenizer(LilaTokenizer *tok);

#endif
