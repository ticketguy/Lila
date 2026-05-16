#ifndef LILA_MODEL_H
#define LILA_MODEL_H

#include <stdint.h>
#include <stddef.h>

/*
 * Lila Model Format
 * 
 * Supports two weight storage modes:
 *   1. FigQuant INT4: codebook + scales + packed indices (custom)
 *   2. Q4_K: raw GGUF Q4_K blocks stored directly (native quality)
 *
 * Q4_K mode stores GGUF blocks as-is — no re-quantization, no precision loss.
 * The engine dequantizes on-the-fly during matmul (same as llama.cpp).
 */

#define LILA_MAGIC 0x4C494C41  /* "LILA" */
#define LILA_VERSION 1
#define LILA_MAX_LAYERS 64
#define LILA_MAX_VOCAB 524288
#define LILA_GROUP_SIZE 128
#define LILA_CODEBOOK_SIZE 16

/* Quantization types for weight tensors */
#define QUANT_NONE      0   /* FP32 raw */
#define QUANT_FIGQUANT  1   /* FigQuant INT4 (codebook + scales + packed) */
#define QUANT_Q4_K      2   /* GGUF Q4_K blocks stored raw */
#define QUANT_Q6_K      3   /* GGUF Q6_K blocks stored raw */
#define QUANT_F16       4   /* FP16 */

/* Quantized weight tensor */
typedef struct {
    uint8_t *data;          /* Raw weight data pointer (into mmap) */
    int quant_type;         /* QUANT_FIGQUANT, QUANT_Q4_K, etc. */
    int rows;
    int cols;
    size_t data_size;       /* Total bytes of weight data */
    
    /* FigQuant-specific fields (only when quant_type == QUANT_FIGQUANT) */
    uint8_t *indices;       /* Packed 4-bit (2 per byte) */
    float codebook[LILA_CODEBOOK_SIZE];  /* 16 dequant values */
    float *scales;          /* Per-group scales (FP32) */
    int n_groups;
} LilaQuantWeight;

/* LoRA adapter (for Memory Fabric) */
typedef struct {
    float *A;           /* [in_features, rank] */
    float *B;           /* [rank, out_features] */
    float gate;         /* Namespace gate value [0,1] */
    int rank;
    int in_features;
    int out_features;
} LilaLoRA;

/* Memory Fabric — 5 namespace adapters per layer */
#define LILA_N_NAMESPACES 5
typedef struct {
    LilaLoRA adapters[LILA_N_NAMESPACES];
} LilaMemoryFabric;

/* Transformer layer */
typedef struct {
    /* Attention */
    LilaQuantWeight q_proj;
    LilaQuantWeight k_proj;
    LilaQuantWeight v_proj;
    LilaQuantWeight o_proj;
    
    /* MLP */
    LilaQuantWeight gate_proj;
    LilaQuantWeight up_proj;
    LilaQuantWeight down_proj;
    
    /* Norms */
    float *input_layernorm;
    float *post_attention_layernorm;
    
    /* Memory Fabric for this layer */
    LilaMemoryFabric fabric;
    
    int hidden_size;
    int intermediate_size;
    int n_heads;
    int n_kv_heads;
    int head_dim;
} LilaLayer;

/* KV Cache */
typedef struct {
    float *key_cache;
    float *value_cache;
    int max_seq_len;
    int current_pos;
} LilaKVCache;

/* Full model */
typedef struct {
    /* Header */
    uint32_t magic;
    uint32_t version;
    
    /* Config */
    int n_layers;
    int hidden_size;
    int intermediate_size;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    float rope_theta;
    float rms_norm_eps;
    int weight_quant_type;      /* Global quant type for this model */
    
    /* Weights */
    float *token_embedding;     /* [vocab_size, hidden_size] FP32 (NULL if quantized) */
    uint8_t *embed_data;        /* Raw quantized embedding (Q6_K blocks) */
    int embed_quant_type;       /* 0=FP32 in token_embedding, 5=Q6_K in embed_data */
    int embed_bytes_per_row;    /* Bytes per vocab row in embed_data */
    LilaLayer layers[LILA_MAX_LAYERS];
    float *final_norm;
    float *lm_head;             /* [vocab_size, hidden_size] or tied */
    
    /* Runtime */
    LilaKVCache kv_cache;
    
    /* Memory map */
    void *mmap_addr;
    size_t mmap_size;
} LilaModel;

/* API */
LilaModel *lila_load_model(const char *path);
void lila_free_model(LilaModel *model);
int lila_generate_token(LilaModel *model, int *tokens, int n_tokens);
void lila_generate(LilaModel *model, int *tokens, int n_tokens, int max_new_tokens,
                   void (*callback)(int token, void *ctx), void *ctx);

/* Weight compute dispatch — calls correct dequant based on quant_type */
void weight_matvec(float *out, const LilaQuantWeight *w, const float *vec);

#endif /* LILA_MODEL_H */
