#ifndef LILA_TOKENIZER_H
#define LILA_TOKENIZER_H

typedef struct LilaTokenizer LilaTokenizer;

LilaTokenizer *lila_load_tokenizer(const char *vocab_path);
const char *lila_decode_token(LilaTokenizer *tok, int token_id);
int lila_encode_char(LilaTokenizer *tok, char c);
void lila_free_tokenizer(LilaTokenizer *tok);

#endif
