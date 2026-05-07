#ifndef LILA_MODEL_H
#define LILA_MODEL_H

#include <stdint.h>
#include <stddef.h>

/*
 * Lila Model Format
 * 
 * Weights stored as FigQuant INT4:
 *   - 16-value codebook per layer (64 bytes)
 *   - Packed 4-bit indices (2 per byte)
 *   - Per-group FP16 scales
 *
 * Memory layout optimized for:
 *   - mmap loading (zero-copy from disk)
 *   - SIMD dequantization (codebook fits in one register)
 *   - Cache-friendly access patterns
 */

#define LILA_MAGIC 0x4C494C41  /* "LILA" */
#define LILA_VERSION 1
#define LILA_MAX_LAYERS 64
#define LILA_MAX_VOCAB 128000
#define LILA_GROUP_SIZE 128
#define LILA_CODEBOOK_SIZE 16

/* Quantized weight tensor */
typedef struct {
    uint8_t *indices;       /* Packed 4-bit (2 per byte) */
    float codebook[LILA_CODEBOOK_SIZE];  /* 16 dequant values */
    uint16_t *scales;       /* Per-group FP16 scales */
    int rows;
    int cols;
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
    /* Namespace indices: 0=personal, 1=episodic, 2=wiki, 3=schedule, 4=contested */
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
    float *input_layernorm;     /* RMSNorm weights */
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
    float *key_cache;       /* [n_layers, max_seq, n_kv_heads, head_dim] */
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
    
    /* Weights */
    float *token_embedding;     /* [vocab_size, hidden_size] */
    LilaLayer layers[LILA_MAX_LAYERS];
    float *final_norm;          /* RMSNorm weights */
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

#endif /* LILA_MODEL_H */
