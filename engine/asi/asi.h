#ifndef LILA_ASI_H
#define LILA_ASI_H

/*
 * .asi — Active System Image
 *
 * A single self-contained file that IS Lila.
 * Contains everything: weights, adapters, tokenizer, compute kernels,
 * harness, personality. Load it and she thinks.
 *
 * Platform independent via portable bytecode (LilaVM).
 * Runs on any CPU with enough RAM.
 */

#include <stdint.h>
#include <stddef.h>

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MAGIC & VERSION                                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define ASI_MAGIC 0x41534921 /* "ASI!" in little-endian */
#define ASI_VERSION 1
#define ASI_PAGE_SIZE 4096 /* Section alignment */

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  SECTION TYPES                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef enum
{
    ASI_SECTION_MODEL_CONFIG = 0x01,
    ASI_SECTION_WEIGHTS = 0x02,
    ASI_SECTION_MEMORY_FABRIC = 0x03,
    ASI_SECTION_TOKENIZER = 0x04,
    ASI_SECTION_BYTECODE = 0x05,
    ASI_SECTION_HARNESS = 0x06,
    ASI_SECTION_PERSONALITY = 0x07,
    ASI_SECTION_METADATA = 0x08,
} AsiSectionType;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  FLAGS                                                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define ASI_FLAG_HAS_FABRIC (1 << 0)
#define ASI_FLAG_HAS_BYTECODE (1 << 1)
#define ASI_FLAG_HAS_HARNESS (1 << 2)
#define ASI_FLAG_HAS_PERSONALITY (1 << 3)
#define ASI_FLAG_QUANT_INT4 (1 << 4)
#define ASI_FLAG_TOKENIZER_BPE (1 << 5)
#define ASI_FLAG_HOT_RELOAD (1 << 6)
#define ASI_FLAG_ENCRYPTED (1 << 7)

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  HEADER (64 bytes, at offset 0)                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint32_t magic;                /* ASI_MAGIC */
    uint32_t version;              /* ASI_VERSION */
    uint32_t flags;                /* ASI_FLAG_* */
    uint32_t n_sections;           /* Number of sections */
    uint64_t total_size;           /* Total file size */
    uint64_t section_table_offset; /* Offset to section table (always 64) */
    uint8_t identity_hash[32];     /* SHA-256 of personality section */
} AsiHeader;

_Static_assert(sizeof(AsiHeader) == 64, "AsiHeader must be 64 bytes");

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  SECTION TABLE ENTRY (32 bytes each)                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint32_t type;     /* AsiSectionType */
    uint32_t flags;    /* Section-specific flags */
    uint64_t offset;   /* Byte offset from file start */
    uint64_t size;     /* Section size in bytes */
    uint64_t checksum; /* CRC-64 of section data */
} AsiSectionEntry;

_Static_assert(sizeof(AsiSectionEntry) == 32, "AsiSectionEntry must be 32 bytes");

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MODEL CONFIG (inside ASI_SECTION_MODEL_CONFIG)                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint32_t n_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t head_dim;
    float rope_theta;
    float rms_norm_eps;
    uint32_t quant_type; /* 0=FP32, 1=FP16, 2=INT8, 3=INT4_FIGQUANT */
    uint32_t group_size; /* Quantization group size */
} AsiModelConfig;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MEMORY FABRIC HEADER                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define ASI_N_NAMESPACES 5
/* Namespace indices: 0=personal, 1=episodic, 2=wiki, 3=schedule, 4=contested */

typedef struct
{
    uint32_t n_namespaces; /* 5 */
    uint32_t n_layers;     /* Must match model */
    uint32_t default_rank; /* Default LoRA rank */
    uint32_t reserved;
} AsiFabricHeader;

/* Per-adapter header (precedes weight data) */
typedef struct
{
    uint32_t rank;
    uint32_t in_features;
    uint32_t out_features;
    float gate; /* Activation [0,1] */
} AsiAdapterHeader;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  TOKENIZER HEADER                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint32_t vocab_size;
    uint32_t n_merges;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t pad_id;
    uint32_t reserved;
} AsiTokenizerHeader;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  BYTECODE HEADER (LilaVM portable kernels)                                */
/* ═══════════════════════════════════════════════════════════════════════════ */

#define LILAVM_VERSION 1

/* Kernel IDs */
typedef enum
{
    KERNEL_MATMUL = 0x01,
    KERNEL_MATVEC = 0x02,
    KERNEL_RMSNORM = 0x03,
    KERNEL_SOFTMAX = 0x04,
    KERNEL_SILU = 0x05,
    KERNEL_ROPE = 0x06,
    KERNEL_DEQUANT_INT4 = 0x07,
    KERNEL_ATTENTION = 0x08,
    KERNEL_LORA_FUSED = 0x09,
    KERNEL_LAYERNORM = 0x0A,
    KERNEL_GELU = 0x0B,
} AsiKernelId;

typedef struct
{
    uint32_t n_kernels;
    uint32_t vm_version; /* LILAVM_VERSION */
    uint32_t reserved[2];
} AsiBytecodeHeader;

typedef struct
{
    uint32_t kernel_id;       /* AsiKernelId */
    uint32_t bytecode_offset; /* Relative to bytecode section data start */
    uint32_t bytecode_size;   /* In bytes */
    uint32_t flags;           /* Hints: 0x01=SIMD-friendly, 0x02=parallelizable */
} AsiKernelEntry;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  HARNESS HEADER                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint32_t n_tools;
    uint32_t n_patterns;
    uint32_t tools_offset;    /* Relative offset to tool defs */
    uint32_t patterns_offset; /* Relative offset to execution patterns */
} AsiHarnessHeader;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PERSONALITY HEADER                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint64_t interactions_count;
    uint64_t last_active;   /* Unix timestamp */
    uint32_t state_dim;     /* Personality vector dimension */
    uint32_t n_identity_kv; /* Number of identity key-value pairs */
} AsiPersonalityHeader;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  METADATA                                                                 */
/* ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    uint64_t created_at;      /* Unix timestamp */
    uint64_t modified_at;     /* Unix timestamp */
    char creator[64];         /* e.g. "Little Fig v1.0" */
    char base_model[64];      /* e.g. "gemma-3-4b-it" */
    uint8_t content_hash[32]; /* SHA-256 of all other sections */
} AsiMetadata;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  RUNTIME API                                                              */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* Opaque runtime handle */
typedef struct AsiRuntime AsiRuntime;

/* Load and boot from .asi file */
AsiRuntime *asi_load(const char *path);

/* Generate next token */
int asi_generate_token(AsiRuntime *rt, int *tokens, int n_tokens);

/* Generate sequence with callback */
void asi_generate(AsiRuntime *rt, int *tokens, int n_tokens, int max_new,
                  void (*on_token)(int token, void *ctx), void *ctx);

/* Tokenize a string into token IDs.
 * Returns number of tokens written into out_ids.
 * Returns 0 if tokenizer not loaded. */
int asi_tokenize(AsiRuntime *rt, const char *text, int *out_ids, int max_ids);

/* Decode a single token ID to its string piece.
 * Returns NULL if tokenizer not loaded or token out of range.
 * Returned pointer is valid for the lifetime of the runtime. */
const char *asi_decode_token(AsiRuntime *rt, int token_id);

/* Hot-reload adapters from another .asi */
int asi_reload_fabric(AsiRuntime *rt, const char *new_asi_path);

/* Get personality state */
const char *asi_get_identity(AsiRuntime *rt, const char *key);

/* Shutdown */
void asi_free(AsiRuntime *rt);

/* Query sections */
int asi_has_section(AsiRuntime *rt, AsiSectionType type);
uint64_t asi_section_size(AsiRuntime *rt, AsiSectionType type);

#endif /* LILA_ASI_H */