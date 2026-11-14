#ifndef LILA_ASI_V2_H
#define LILA_ASI_V2_H

/*
 * ASI v2 — Active System Image (Portable Inference Cartridge)
 *
 * Self-describing, page-aligned, mmap-ready binary format separating:
 *   - model.asi (cartridge)
 *   - Lila ASI Runtime (machine-direct C/ASM engine)
 *   - Lila Host Application (Core, Harness, OIA, vTalks, Island)
 */

#include <stdint.h>
#include <stddef.h>

#define ASI_V2_MAGIC     0x41534921u /* "ASI!" little-endian */
#define ASI_V2_VERSION   2u
#define ASI_V2_PAGE_SIZE 4096u

typedef enum {
    ASI_ARCH_UNKNOWN        = 0,
    ASI_ARCH_DENSE_LLAMA    = 1,
    ASI_ARCH_HYBRID_QWEN35  = 2,
    ASI_ARCH_DENSE_GEMMA    = 3,
    ASI_ARCH_DENSE_QWEN2    = 4,
    ASI_ARCH_DENSE_QWEN3    = 5,
} AsiV2ArchId;

typedef enum {
    ASI_SEC_MODEL_CONFIG  = 0x01,
    ASI_SEC_WEIGHTS       = 0x02,
    ASI_SEC_MEMORY_FABRIC = 0x03, /* Reserved for future Ember integration */
    ASI_SEC_TOKENIZER     = 0x04,
    ASI_SEC_BYTECODE_LEGACY = 0x05, /* Untrusted/ignored in v2 load path */
    ASI_SEC_HARNESS_LEGACY  = 0x06, /* Live Harness owns tools/permissions */
    ASI_SEC_PERSONALITY   = 0x07,
    ASI_SEC_METADATA      = 0x08,
    ASI_SEC_CHAT_TEMPLATE = 0x09,
    ASI_SEC_VISION        = 0x0A,
    ASI_SEC_AUDIO         = 0x0B,
    ASI_SEC_ADAPTERS      = 0x0C,
    ASI_SEC_RUNTIME_HINTS = 0x0D,
    ASI_SEC_SIGNATURE     = 0x0E,
} AsiV2SectionType;

typedef enum {
    ASI_QUANT_F32     = 0,
    ASI_QUANT_FIGINT4 = 1,
    ASI_QUANT_Q4_K    = 2,
    ASI_QUANT_Q6_K    = 3,
    ASI_QUANT_F16     = 4,
    ASI_QUANT_Q8_0    = 5,
} AsiV2QuantType;

#define ASI_TENSOR_LAYOUT_ROW_MAJOR   (1u << 0)
#define ASI_TENSOR_LAYOUT_AVX2_REPACK (1u << 1)

#define ASI_V2_FLAG_HAS_VISION        (1u << 0)
#define ASI_V2_FLAG_HAS_AUDIO         (1u << 1)
#define ASI_V2_FLAG_HAS_ADAPTERS      (1u << 2)
#define ASI_V2_FLAG_HAS_SIGNATURE     (1u << 3)
#define ASI_V2_FLAG_HAS_CHAT_TEMPLATE (1u << 4)

/* 64-byte file header */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t arch_id;
    uint32_t flags;
    uint32_t n_sections;
    uint32_t reserved;
    uint64_t total_size;
    uint64_t section_table_offset;
    uint8_t  identity_hash[24];
} AsiV2Header;

_Static_assert(sizeof(AsiV2Header) == 64, "AsiV2Header must be 64 bytes");

/* 32-byte section directory entry */
typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;
    uint64_t checksum; /* CRC-64 ECMA */
} AsiV2SectionEntry;

_Static_assert(sizeof(AsiV2SectionEntry) == 32, "AsiV2SectionEntry must be 32 bytes");

/* 96-byte explicit architecture configuration */
typedef struct {
    uint32_t arch_id;
    uint32_t n_layers;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t rope_dim;
    float    rope_theta;
    float    rms_norm_eps;
    float    embed_scale;
    uint32_t full_attention_interval; /* e.g. 4 for Qwen3.5, 1 for dense */
    uint32_t ssm_conv_kernel;
    uint32_t ssm_state_size;
    uint32_t ssm_group_count;
    uint32_t ssm_inner_size;
    uint32_t ssm_time_step_rank;
    uint32_t default_quant_type;
    uint32_t reserved[4];
} AsiV2ModelConfig;

_Static_assert(sizeof(AsiV2ModelConfig) == 96, "AsiV2ModelConfig must be 96 bytes");

/* 72-byte weights directory header + tensor table inside ASI_SEC_WEIGHTS */
typedef struct {
    uint32_t n_tensors;
    uint32_t flags;
    uint64_t data_offset; /* Relative to start of ASI_SEC_WEIGHTS */
} AsiV2WeightsHeader;

_Static_assert(sizeof(AsiV2WeightsHeader) == 16, "AsiV2WeightsHeader must be 16 bytes");

typedef struct {
    char     name[40];
    uint32_t quant_type;
    uint32_t layout_flags;
    uint64_t shape[2];
    uint64_t data_offset; /* Relative to ASI_SEC_WEIGHTS + WeightsHeader.data_offset */
    uint64_t data_size;
} AsiV2TensorEntry;

_Static_assert(sizeof(AsiV2TensorEntry) == 80, "AsiV2TensorEntry must be 80 bytes");

typedef struct {
    const uint8_t           *base;
    size_t                   file_size;
    int                      owns_mmap;
    AsiV2Header              header;
    const AsiV2SectionEntry *sections;
    AsiV2ModelConfig         config;
    const AsiV2TensorEntry  *tensors;
    uint32_t                 n_tensors;
    const uint8_t           *weights_data;
    size_t                   weights_data_size;
} AsiV2Cartridge;

uint64_t asi_v2_crc64(const uint8_t *data, size_t len);
int asi_v2_validate_memory(const uint8_t *data, size_t size, AsiV2Cartridge *out,
                           char *err_buf, size_t err_cap);
int asi_v2_open(const char *path, AsiV2Cartridge *out, char *err_buf, size_t err_cap);
void asi_v2_close(AsiV2Cartridge *cart);
const AsiV2SectionEntry *asi_v2_find_section(const AsiV2Cartridge *cart, uint32_t type);
const AsiV2TensorEntry *asi_v2_find_tensor(const AsiV2Cartridge *cart, const char *name);
const char *asi_v2_arch_name(uint32_t arch_id);

#endif /* LILA_ASI_V2_H */
