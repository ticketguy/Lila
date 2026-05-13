#include "asi.h"
#include "lilavm.h"
#include "../runtime/model.h"
#include "../runtime/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PLATFORM ABSTRACTION — mmap on Linux/macOS, MapViewOfFile on Windows     */
/* ═══════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>
#include <io.h>

static void *platform_mmap(const char *path, size_t *out_size)
{
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
        return NULL;

    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    *out_size = (size_t)li.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap)
    {
        CloseHandle(hFile);
        return NULL;
    }

    void *mapped = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    return mapped;
}

static void platform_munmap(void *addr, size_t size)
{
    (void)size;
    if (addr)
        UnmapViewOfFile(addr);
}

static void platform_madvise_seq(void *addr, size_t size)
{
    (void)addr;
    (void)size;
}
static void platform_madvise_rand(void *addr, size_t size)
{
    (void)addr;
    (void)size;
}

/* Unused on current code path — suppress warnings */
static const char *platform_tmp_vocab(void) __attribute__((unused));
static const char *platform_tmp_vocab(void)
{
    static char path[MAX_PATH];
    GetTempPathA(MAX_PATH, path);
    strcat(path, ".lila_asi_vocab.tmp");
    return path;
}
static void platform_unlink(const char *path) __attribute__((unused));
static void platform_unlink(const char *path) { DeleteFileA(path); }

#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static void *platform_mmap(const char *path, size_t *out_size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NULL;
    struct stat st;
    fstat(fd, &st);
    *out_size = st.st_size;
    void *mapped = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED)
        return NULL;
    return mapped;
}

static void platform_munmap(void *addr, size_t size)
{
    if (addr)
        munmap(addr, size);
}

static void platform_madvise_seq(void *addr, size_t size)
{
    madvise(addr, size, MADV_SEQUENTIAL);
}
static void platform_madvise_rand(void *addr, size_t size)
{
    madvise(addr, size, MADV_RANDOM);
}

static const char *platform_tmp_vocab(void) __attribute__((unused));
static const char *platform_tmp_vocab(void) { return "/tmp/.lila_asi_vocab.tmp"; }
static void platform_unlink(const char *path) __attribute__((unused));
static void platform_unlink(const char *path) { unlink(path); }
#endif

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  RUNTIME STRUCT (opaque to user)                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */

struct AsiRuntime
{
    void *mmap_addr;
    size_t mmap_size;

    AsiHeader header;
    AsiSectionEntry *sections;

    AsiModelConfig config;
    LilaModel *model;

    LilaTokenizer *tokenizer;

    LilaVM vm;
    uint32_t *kernels[16];
    int kernel_lens[16];
    int use_native;

    float *fabric_data;
    char *identity_kv;
    uint64_t interactions;

    int booted;
};

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  SECTION LOOKUP                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

static AsiSectionEntry *find_section(AsiRuntime *rt, AsiSectionType type)
{
    for (uint32_t i = 0; i < rt->header.n_sections; i++)
    {
        if (rt->sections[i].type == (uint32_t)type)
            return &rt->sections[i];
    }
    return NULL;
}

static void *section_data(AsiRuntime *rt, AsiSectionEntry *sec)
{
    if (!sec)
        return NULL;
    return (uint8_t *)rt->mmap_addr + sec->offset;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  LOAD MODEL CONFIG                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int load_model_config(AsiRuntime *rt)
{
    AsiSectionEntry *sec = find_section(rt, ASI_SECTION_MODEL_CONFIG);
    if (!sec)
    {
        fprintf(stderr, "ASI: Missing MODEL_CONFIG section\n");
        return -1;
    }

    memcpy(&rt->config, section_data(rt, sec), sizeof(AsiModelConfig));

    fprintf(stderr, "ASI: Model config loaded\n");
    fprintf(stderr, "     Layers: %u, Hidden: %u, Vocab: %u\n",
            rt->config.n_layers, rt->config.hidden_size, rt->config.vocab_size);
    fprintf(stderr, "     Heads: %u, KV Heads: %u, Seq: %u\n",
            rt->config.n_heads, rt->config.n_kv_heads, rt->config.max_seq_len);
    fprintf(stderr, "     Quant: %s, Group: %u\n",
            rt->config.quant_type == 3 ? "INT4_FigQuant" : "other",
            rt->config.group_size);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  LOAD WEIGHTS (zero-copy via mmap)                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int load_weights(AsiRuntime *rt)
{
    AsiSectionEntry *sec = find_section(rt, ASI_SECTION_WEIGHTS);
    if (!sec)
    {
        fprintf(stderr, "ASI: Missing WEIGHTS section\n");
        return -1;
    }

    rt->model = calloc(1, sizeof(LilaModel));
    rt->model->n_layers = rt->config.n_layers;
    rt->model->hidden_size = rt->config.hidden_size;
    rt->model->intermediate_size = rt->config.intermediate_size;
    rt->model->n_heads = rt->config.n_heads;
    rt->model->n_kv_heads = rt->config.n_kv_heads;
    rt->model->head_dim = rt->config.hidden_size / rt->config.n_heads;
    rt->model->vocab_size = rt->config.vocab_size;
    rt->model->max_seq_len = rt->config.max_seq_len;
    rt->model->rope_theta = rt->config.rope_theta;
    rt->model->rms_norm_eps = rt->config.rms_norm_eps;

    rt->model->mmap_addr = section_data(rt, sec);
    rt->model->mmap_size = sec->size;

    uint8_t *ptr = (uint8_t *)rt->model->mmap_addr;
    uint8_t *weights_end = ptr + sec->size;

    int global_quant = rt->config.quant_type;
    int group_size = rt->config.group_size > 0 ? rt->config.group_size : 128;

    rt->model->weight_quant_type = global_quant;

    /* Token embedding (FP32) */
    size_t embed_size = (size_t)rt->config.vocab_size * rt->config.hidden_size * sizeof(float);
    rt->model->token_embedding = (float *)ptr;
    ptr += embed_size;

    for (int layer_idx = 0; layer_idx < rt->config.n_layers; layer_idx++)
    {
        if (ptr >= weights_end)
            break;

        LilaLayer *layer = &rt->model->layers[layer_idx];
        layer->hidden_size = rt->config.hidden_size;
        layer->intermediate_size = rt->config.intermediate_size;
        layer->n_heads = rt->config.n_heads;
        layer->n_kv_heads = rt->config.n_kv_heads;
        layer->head_dim = rt->config.hidden_size / rt->config.n_heads;

        LilaQuantWeight *projs[7] = {
            &layer->q_proj, &layer->k_proj, &layer->v_proj, &layer->o_proj,
            &layer->gate_proj, &layer->up_proj, &layer->down_proj};

        for (int p = 0; p < 7; p++)
        {
            if (ptr + 12 > weights_end)
                break;

            int rows, cols, quant_type;
            memcpy(&rows, ptr, 4);
            ptr += 4;
            memcpy(&cols, ptr, 4);
            ptr += 4;
            memcpy(&quant_type, ptr, 4);
            ptr += 4;

            projs[p]->rows = rows;
            projs[p]->cols = cols;
            projs[p]->quant_type = quant_type;

            if (rows == 0 || cols == 0)
                continue;

            if (quant_type == QUANT_Q4_K)
            {
                int n_elements = rows * cols;
                int n_blocks = (n_elements + 255) / 256;
                size_t data_size = (size_t)n_blocks * 144;
                if (ptr + data_size > weights_end)
                    break;
                projs[p]->data = ptr;
                projs[p]->data_size = data_size;
                ptr += data_size;
            }
            else if (quant_type == QUANT_Q6_K)
            {
                int n_elements = rows * cols;
                int n_blocks = (n_elements + 255) / 256;
                size_t data_size = (size_t)n_blocks * 210;
                if (ptr + data_size > weights_end)
                    break;
                projs[p]->data = ptr;
                projs[p]->data_size = data_size;
                ptr += data_size;
            }
            else if (quant_type == QUANT_FIGQUANT)
            {
                if (ptr + 64 > weights_end)
                    break;
                memcpy(projs[p]->codebook, ptr, 16 * sizeof(float));
                ptr += 64;

                int n_elements = rows * cols;
                int n_groups = (n_elements + group_size - 1) / group_size;
                size_t scales_size = n_groups * sizeof(float);
                if (ptr + scales_size > weights_end)
                    break;
                projs[p]->scales = (float *)ptr;
                ptr += scales_size;
                projs[p]->n_groups = n_groups;

                size_t packed_size = (n_elements + 1) / 2;
                if (ptr + packed_size > weights_end)
                    break;
                projs[p]->indices = ptr;
                projs[p]->data = ptr;
                ptr += packed_size;
            }
            else
            {
                break;
            }
        }

        size_t norm_size = rt->config.hidden_size * sizeof(float);
        if (ptr + norm_size <= weights_end)
        {
            layer->input_layernorm = (float *)ptr;
            ptr += norm_size;
        }
        if (ptr + norm_size <= weights_end)
        {
            layer->post_attention_layernorm = (float *)ptr;
            ptr += norm_size;
        }
    }

    /* Final norm */
    size_t norm_size = rt->config.hidden_size * sizeof(float);
    if (ptr + norm_size <= weights_end)
    {
        rt->model->final_norm = (float *)ptr;
        ptr += norm_size;
    }

    /* LM Head */
    if (ptr + 4 <= weights_end)
    {
        uint32_t flag;
        memcpy(&flag, ptr, 4);
        if (flag == 0xFFFFFFFF)
        {
            rt->model->lm_head = rt->model->token_embedding;
            ptr += 4;
        }
        else
        {
            size_t lm_size = (size_t)rt->config.vocab_size * rt->config.hidden_size * sizeof(float);
            if (ptr + lm_size <= weights_end)
            {
                rt->model->lm_head = (float *)ptr;
                ptr += lm_size;
            }
        }
    }

    /* Fixed format strings for MinGW (%llu for size_t, %u for uint32) */
    fprintf(stderr, "ASI: Weights fully parsed (%llu MB, %u layers wired)\n",
            (unsigned long long)(sec->size / (1024 * 1024)),
            rt->config.n_layers);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  LOAD MEMORY FABRIC                                                       */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int load_memory_fabric(AsiRuntime *rt)
{
    AsiSectionEntry *sec = find_section(rt, ASI_SECTION_MEMORY_FABRIC);
    if (!sec)
    {
        fprintf(stderr, "ASI: No Memory Fabric section (base model only)\n");
        return 0;
    }

    uint8_t *ptr = (uint8_t *)section_data(rt, sec);
    AsiFabricHeader *fhdr = (AsiFabricHeader *)ptr;
    ptr += sizeof(AsiFabricHeader);

    fprintf(stderr, "ASI: Memory Fabric loaded\n");
    fprintf(stderr, "     Namespaces: %u, Layers: %u, Default rank: %u\n",
            fhdr->n_namespaces, fhdr->n_layers, fhdr->default_rank);

    for (uint32_t layer = 0; layer < fhdr->n_layers && layer < (uint32_t)rt->config.n_layers; layer++)
    {
        for (uint32_t ns = 0; ns < fhdr->n_namespaces && ns < ASI_N_NAMESPACES; ns++)
        {
            AsiAdapterHeader *ahdr = (AsiAdapterHeader *)ptr;
            ptr += sizeof(AsiAdapterHeader);

            LilaLoRA *adapter = &rt->model->layers[layer].fabric.adapters[ns];
            adapter->rank = ahdr->rank;
            adapter->in_features = ahdr->in_features;
            adapter->out_features = ahdr->out_features;
            adapter->gate = ahdr->gate;

            if (ahdr->rank > 0)
            {
                size_t a_size = (size_t)ahdr->in_features * ahdr->rank * sizeof(float);
                size_t b_size = (size_t)ahdr->rank * ahdr->out_features * sizeof(float);
                adapter->A = (float *)ptr;
                ptr += a_size;
                adapter->B = (float *)ptr;
                ptr += b_size;
            }
        }
    }

    fprintf(stderr, "     Adapters active: personal, episodic, wiki, schedule, contested\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  LOAD TOKENIZER                                                           */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int load_tokenizer(AsiRuntime *rt)
{
    AsiSectionEntry *sec = find_section(rt, ASI_SECTION_TOKENIZER);
    if (!sec)
    {
        fprintf(stderr, "ASI: No tokenizer section\n");
        return -1;
    }

    uint8_t *ptr = (uint8_t *)section_data(rt, sec);
    AsiTokenizerHeader *thdr = (AsiTokenizerHeader *)ptr;
    ptr += sizeof(AsiTokenizerHeader);

    size_t remaining = sec->size - sizeof(AsiTokenizerHeader);
    rt->tokenizer = lila_load_tokenizer_from_memory(
        ptr, remaining,
        thdr->vocab_size,
        thdr->bos_id,
        thdr->eos_id,
        thdr->pad_id);

    if (!rt->tokenizer)
        fprintf(stderr, "ASI: Tokenizer extraction failed (continuing without)\n");

    fprintf(stderr, "ASI: Tokenizer loaded (vocab=%u, merges=%u)\n",
            thdr->vocab_size, thdr->n_merges);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  LOAD BYTECODE                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int load_bytecode(AsiRuntime *rt)
{
    AsiSectionEntry *sec = find_section(rt, ASI_SECTION_BYTECODE);
    if (!sec)
    {
        fprintf(stderr, "ASI: No bytecode section — using native kernels only\n");
        rt->use_native = 1;
        return 0;
    }

    uint8_t *ptr = (uint8_t *)section_data(rt, sec);
    AsiBytecodeHeader *bhdr = (AsiBytecodeHeader *)ptr;
    ptr += sizeof(AsiBytecodeHeader);

    fprintf(stderr, "ASI: Bytecode loaded (VM v%u, %u kernels)\n",
            bhdr->vm_version, bhdr->n_kernels);

    AsiKernelEntry *entries = (AsiKernelEntry *)ptr;
    ptr += bhdr->n_kernels * sizeof(AsiKernelEntry);
    uint8_t *code_base = ptr;

    for (uint32_t i = 0; i < bhdr->n_kernels && i < 16; i++)
    {
        AsiKernelEntry *ke = &entries[i];
        rt->kernels[ke->kernel_id] = (uint32_t *)(code_base + ke->bytecode_offset);
        rt->kernel_lens[ke->kernel_id] = ke->bytecode_size / 4;
        fprintf(stderr, "     Kernel 0x%02X: %u instructions\n",
                ke->kernel_id, rt->kernel_lens[ke->kernel_id]);
    }

    vm_init(&rt->vm);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  LOAD PERSONALITY                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

static int load_personality(AsiRuntime *rt)
{
    AsiSectionEntry *sec = find_section(rt, ASI_SECTION_PERSONALITY);
    if (!sec)
    {
        fprintf(stderr, "ASI: No personality section (fresh instance)\n");
        return 0;
    }

    AsiPersonalityHeader *phdr = (AsiPersonalityHeader *)section_data(rt, sec);
    rt->interactions = phdr->interactions_count;

    fprintf(stderr, "ASI: Personality loaded\n");
    fprintf(stderr, "     Interactions: %llu, State dim: %u\n",
            (unsigned long long)phdr->interactions_count, phdr->state_dim);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API: LOAD                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

AsiRuntime *asi_load(const char *path)
{
    fprintf(stderr, "\n🌸 Loading ASI: %s\n", path);

    size_t file_size = 0;
    void *mapped = platform_mmap(path, &file_size);
    if (!mapped)
    {
        fprintf(stderr, "ASI: Cannot open/map file: %s\n", path);
        return NULL;
    }

    if (file_size < sizeof(AsiHeader))
    {
        fprintf(stderr, "ASI: File too small (%llu bytes)\n", (unsigned long long)file_size);
        platform_munmap(mapped, file_size);
        return NULL;
    }

    platform_madvise_seq(mapped, file_size);

    AsiRuntime *rt = calloc(1, sizeof(AsiRuntime));
    rt->mmap_addr = mapped;
    rt->mmap_size = file_size;

    memcpy(&rt->header, mapped, sizeof(AsiHeader));

    if (rt->header.magic != ASI_MAGIC)
    {
        fprintf(stderr, "ASI: Invalid magic: 0x%08X (expected 0x%08X)\n",
                rt->header.magic, ASI_MAGIC);
        asi_free(rt);
        return NULL;
    }

    if (rt->header.version > ASI_VERSION)
    {
        fprintf(stderr, "ASI: Unsupported version: %u (max: %u)\n",
                rt->header.version, ASI_VERSION);
        asi_free(rt);
        return NULL;
    }

    fprintf(stderr, "ASI: Header OK (v%u, %u sections, %llu bytes)\n",
            rt->header.version, rt->header.n_sections,
            (unsigned long long)file_size);

    rt->sections = (AsiSectionEntry *)((uint8_t *)mapped + rt->header.section_table_offset);

    if (load_model_config(rt) < 0)
    {
        asi_free(rt);
        return NULL;
    }
    if (load_weights(rt) < 0)
    {
        asi_free(rt);
        return NULL;
    }
    load_memory_fabric(rt);
    if (load_tokenizer(rt) < 0)
    {
        asi_free(rt);
        return NULL;
    }
    load_bytecode(rt);
    load_personality(rt);

    extern void lila_init_kv_cache(LilaKVCache * cache, int n_layers, int max_seq,
                                   int n_kv_heads, int head_dim);
    lila_init_kv_cache(&rt->model->kv_cache,
                       rt->config.n_layers,
                       rt->config.max_seq_len,
                       rt->config.n_kv_heads,
                       rt->config.hidden_size / rt->config.n_heads);

    platform_madvise_rand(mapped, file_size);

    rt->booted = 1;
    fprintf(stderr, "\n🌸 ASI loaded. Lila is awake.\n\n");
    return rt;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API: GENERATE                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

int asi_generate_token(AsiRuntime *rt, int *tokens, int n_tokens)
{
    if (!rt || !rt->booted || !rt->model)
        return -1;

    if (!rt->model->token_embedding)
    {
        fprintf(stderr, "ASI: Weights not fully parsed (embedding missing)\n");
        return -1;
    }

    if (rt->model->layers[0].q_proj.data == NULL &&
        rt->model->layers[0].q_proj.indices == NULL)
    {
        fprintf(stderr, "ASI: Layer 0 q_proj not loaded\n");
        return 0;
    }

    fprintf(stderr, "[DBG] generate_token: n_tokens=%d, last_token=%d\n",
            n_tokens, tokens[n_tokens - 1]);

    extern int lila_forward(LilaModel * model, int token, int position);
    int result = lila_forward(rt->model, tokens[n_tokens - 1], n_tokens - 1);

    fprintf(stderr, "[DBG] lila_forward returned: %d\n", result);

    return result;
}

void asi_generate(AsiRuntime *rt, int *tokens, int n_tokens, int max_new,
                  void (*on_token)(int token, void *ctx), void *ctx)
{
    if (!rt || !rt->booted)
        return;
    for (int i = 0; i < max_new; i++)
    {
        int next = asi_generate_token(rt, tokens, n_tokens + i);
        tokens[n_tokens + i] = next;
        if (on_token)
            on_token(next, ctx);
        if (next == 0 || next == -1)
            break;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API: TOKENIZE / DECODE                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */

int asi_tokenize(AsiRuntime *rt, const char *text, int *out_ids, int max_ids)
{
    if (!rt || !rt->tokenizer || !text || !out_ids || max_ids <= 0)
        return 0;
    return lila_encode(rt->tokenizer, text, out_ids, max_ids);
}

const char *asi_decode_token(AsiRuntime *rt, int token_id)
{
    if (!rt || !rt->tokenizer)
        return NULL;
    return lila_decode_token(rt->tokenizer, token_id);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API: HOT RELOAD                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */

int asi_reload_fabric(AsiRuntime *rt, const char *new_asi_path)
{
    if (!rt)
        return -1;

    fprintf(stderr, "ASI: Hot-reloading adapters from %s\n", new_asi_path);

    size_t new_size = 0;
    void *new_map = platform_mmap(new_asi_path, &new_size);
    if (!new_map)
        return -1;

    AsiHeader *new_hdr = (AsiHeader *)new_map;
    if (new_hdr->magic != ASI_MAGIC)
    {
        platform_munmap(new_map, new_size);
        return -1;
    }

    AsiSectionEntry *new_sections = (AsiSectionEntry *)((uint8_t *)new_map + new_hdr->section_table_offset);
    AsiSectionEntry *fab_sec = NULL;

    for (uint32_t i = 0; i < new_hdr->n_sections; i++)
    {
        if (new_sections[i].type == ASI_SECTION_MEMORY_FABRIC)
        {
            fab_sec = &new_sections[i];
            break;
        }
    }

    if (!fab_sec)
    {
        fprintf(stderr, "ASI: New file has no Memory Fabric section\n");
        platform_munmap(new_map, new_size);
        return -1;
    }

    void *old_map = rt->mmap_addr;
    size_t old_size = rt->mmap_size;

    rt->mmap_addr = new_map;
    rt->mmap_size = new_size;
    rt->sections = new_sections;

    load_memory_fabric(rt);

    platform_munmap(old_map, old_size);

    fprintf(stderr, "ASI: Hot-reload complete. New adapters active.\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API: QUERY                                                        */
/* ═══════════════════════════════════════════════════════════════════════════ */

int asi_has_section(AsiRuntime *rt, AsiSectionType type)
{
    return find_section(rt, type) != NULL;
}

uint64_t asi_section_size(AsiRuntime *rt, AsiSectionType type)
{
    AsiSectionEntry *sec = find_section(rt, type);
    return sec ? sec->size : 0;
}

const char *asi_get_identity(AsiRuntime *rt, const char *key)
{
    (void)key;
    if (!rt || !rt->identity_kv)
        return NULL;
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PUBLIC API: FREE                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */

void asi_free(AsiRuntime *rt)
{
    if (!rt)
        return;

    if (rt->model)
    {
        free(rt->model->kv_cache.key_cache);
        free(rt->model->kv_cache.value_cache);
        free(rt->model);
    }

    if (rt->tokenizer)
        lila_free_tokenizer(rt->tokenizer);

    if (rt->mmap_addr)
        platform_munmap(rt->mmap_addr, rt->mmap_size);

    free(rt);
}
