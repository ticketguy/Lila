#include "asi_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

static void set_err(char *buf, size_t cap, const char *msg)
{
    if (!buf || cap == 0)
        return;
    snprintf(buf, cap, "%s", msg ? msg : "unknown error");
}

static uint64_t g_crc64_table[256];
static int g_crc64_table_init = 0;

static void init_crc64_table(void)
{
    for (uint64_t i = 0; i < 256; i++)
    {
        uint64_t crc = i << 56;
        for (int b = 0; b < 8; b++)
        {
            if (crc & 0x8000000000000000ULL)
                crc = (crc << 1) ^ 0x42F0E1EBA9EA3693ULL;
            else
                crc <<= 1;
        }
        g_crc64_table[i] = crc;
    }
    g_crc64_table_init = 1;
}

uint64_t asi_v2_crc64(const uint8_t *data, size_t len)
{
    if (!g_crc64_table_init)
        init_crc64_table();
    uint64_t crc = 0;
    for (size_t i = 0; i < len; i++)
    {
        crc = (crc << 8) ^ g_crc64_table[((crc >> 56) ^ data[i]) & 0xFF];
    }
    return crc;
}

const char *asi_v2_arch_name(uint32_t arch_id)
{
    switch (arch_id)
    {
    case ASI_ARCH_DENSE_LLAMA:   return "dense_llama";
    case ASI_ARCH_HYBRID_QWEN35: return "hybrid_qwen35";
    case ASI_ARCH_DENSE_GEMMA:   return "dense_gemma";
    case ASI_ARCH_DENSE_QWEN2:   return "dense_qwen2";
    case ASI_ARCH_DENSE_QWEN3:   return "dense_qwen3";
    default:                     return "unknown";
    }
}

const AsiV2SectionEntry *asi_v2_find_section(const AsiV2Cartridge *cart, uint32_t type)
{
    if (!cart || !cart->sections)
        return NULL;
    for (uint32_t i = 0; i < cart->header.n_sections; i++)
    {
        if (cart->sections[i].type == type)
            return &cart->sections[i];
    }
    return NULL;
}

const AsiV2TensorEntry *asi_v2_find_tensor(const AsiV2Cartridge *cart, const char *name)
{
    if (!cart || !cart->tensors || !name)
        return NULL;
    for (uint32_t i = 0; i < cart->n_tensors; i++)
    {
        if (strncmp(cart->tensors[i].name, name, sizeof(cart->tensors[i].name)) == 0)
            return &cart->tensors[i];
    }
    return NULL;
}

int asi_v2_validate_memory(const uint8_t *data, size_t size, AsiV2Cartridge *out,
                           char *err_buf, size_t err_cap)
{
    if (!data || !out)
    {
        set_err(err_buf, err_cap, "null cartridge pointer");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->base = data;
    out->file_size = size;

    if (size < sizeof(AsiV2Header))
    {
        set_err(err_buf, err_cap, "file smaller than ASI v2 header");
        return -1;
    }

    memcpy(&out->header, data, sizeof(AsiV2Header));
    if (out->header.magic != ASI_V2_MAGIC)
    {
        set_err(err_buf, err_cap, "invalid ASI magic");
        return -1;
    }
    if (out->header.version != ASI_V2_VERSION)
    {
        set_err(err_buf, err_cap, "unsupported ASI version (expected v2)");
        return -1;
    }
    if (out->header.total_size > size)
    {
        set_err(err_buf, err_cap, "truncated ASI v2 file");
        return -1;
    }
    if (out->header.n_sections == 0 || out->header.n_sections > 64)
    {
        set_err(err_buf, err_cap, "invalid section count");
        return -1;
    }

    uint64_t tbl_off = out->header.section_table_offset;
    uint64_t tbl_bytes = (uint64_t)out->header.n_sections * sizeof(AsiV2SectionEntry);
    if (tbl_off < sizeof(AsiV2Header) || tbl_off + tbl_bytes > size)
    {
        set_err(err_buf, err_cap, "section table out of bounds");
        return -1;
    }

    out->sections = (const AsiV2SectionEntry *)(data + tbl_off);

    for (uint32_t i = 0; i < out->header.n_sections; i++)
    {
        const AsiV2SectionEntry *s = &out->sections[i];
        if (s->offset > size || s->size > size - s->offset)
        {
            set_err(err_buf, err_cap, "section out of file bounds");
            return -1;
        }
        if ((s->offset % ASI_V2_PAGE_SIZE) != 0)
        {
            set_err(err_buf, err_cap, "section offset is not page-aligned");
            return -1;
        }
        int is_bulk_blob = (s->type == ASI_SEC_WEIGHTS || s->type == ASI_SEC_VISION || s->type == ASI_SEC_AUDIO);
        const char *verify_env = getenv("LILA_ASI_VERIFY_CRC");
        if (!is_bulk_blob || (verify_env && verify_env[0] == '1'))
        {
            uint64_t actual_crc = asi_v2_crc64(data + s->offset, (size_t)s->size);
            if (actual_crc != s->checksum)
            {
                set_err(err_buf, err_cap, "section CRC-64 checksum mismatch");
                return -1;
            }
        }
    }

    const AsiV2SectionEntry *cfg_sec = asi_v2_find_section(out, ASI_SEC_MODEL_CONFIG);
    if (!cfg_sec || cfg_sec->size < sizeof(AsiV2ModelConfig))
    {
        set_err(err_buf, err_cap, "missing or invalid MODEL_CONFIG section");
        return -1;
    }
    memcpy(&out->config, data + cfg_sec->offset, sizeof(AsiV2ModelConfig));
    if (out->config.n_layers == 0 || out->config.hidden_size == 0 || out->config.vocab_size == 0)
    {
        set_err(err_buf, err_cap, "MODEL_CONFIG has zero dimensions");
        return -1;
    }

    const AsiV2SectionEntry *w_sec = asi_v2_find_section(out, ASI_SEC_WEIGHTS);
    if (!w_sec || w_sec->size < sizeof(AsiV2WeightsHeader))
    {
        set_err(err_buf, err_cap, "missing or invalid WEIGHTS section");
        return -1;
    }
    AsiV2WeightsHeader wh;
    memcpy(&wh, data + w_sec->offset, sizeof(wh));
    uint64_t dir_bytes = sizeof(AsiV2WeightsHeader) + (uint64_t)wh.n_tensors * sizeof(AsiV2TensorEntry);
    if (wh.data_offset < dir_bytes || wh.data_offset > w_sec->size)
    {
        set_err(err_buf, err_cap, "WEIGHTS tensor directory out of bounds");
        return -1;
    }

    out->n_tensors = wh.n_tensors;
    out->tensors = (const AsiV2TensorEntry *)(data + w_sec->offset + sizeof(AsiV2WeightsHeader));
    out->weights_data = data + w_sec->offset + wh.data_offset;
    out->weights_data_size = (size_t)(w_sec->size - wh.data_offset);

    for (uint32_t i = 0; i < out->n_tensors; i++)
    {
        const AsiV2TensorEntry *t = &out->tensors[i];
        if (t->data_offset > out->weights_data_size ||
            t->data_size > out->weights_data_size - t->data_offset)
        {
            set_err(err_buf, err_cap, "tensor payload out of WEIGHTS bounds");
            return -1;
        }
    }

    if (!asi_v2_find_section(out, ASI_SEC_TOKENIZER))
    {
        set_err(err_buf, err_cap, "missing essential TOKENIZER section");
        return -1;
    }

    return 0;
}

int asi_v2_open(const char *path, AsiV2Cartridge *out, char *err_buf, size_t err_cap)
{
    if (!path || !out)
    {
        set_err(err_buf, err_cap, "invalid arguments");
        return -1;
    }
    size_t file_size = 0;
    const uint8_t *mapped = NULL;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        set_err(err_buf, err_cap, "cannot open ASI v2 file");
        return -1;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    file_size = (size_t)li.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMap)
    {
        set_err(err_buf, err_cap, "CreateFileMappingA failed");
        return -1;
    }
    mapped = (const uint8_t *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        set_err(err_buf, err_cap, "cannot open ASI v2 file");
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        close(fd);
        set_err(err_buf, err_cap, "fstat failed");
        return -1;
    }
    file_size = (size_t)st.st_size;
    void *m = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED)
    {
        set_err(err_buf, err_cap, "mmap failed");
        return -1;
    }
    mapped = (const uint8_t *)m;
#endif

    if (!mapped)
    {
        set_err(err_buf, err_cap, "mapping returned NULL");
        return -1;
    }
    int rc = asi_v2_validate_memory(mapped, file_size, out, err_buf, err_cap);
    out->owns_mmap = 1;
    if (rc != 0)
        asi_v2_close(out);
    return rc;
}

void asi_v2_close(AsiV2Cartridge *cart)
{
    if (!cart)
        return;
    if (cart->owns_mmap && cart->base)
    {
#ifdef _WIN32
        UnmapViewOfFile((LPCVOID)cart->base);
#else
        munmap((void *)cart->base, cart->file_size);
#endif
    }
    memset(cart, 0, sizeof(*cart));
}

static float repack_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t res = 0;
    if (exp == 0)
    {
        if (mant != 0)
        {
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            exp++;
            mant &= ~0x400u;
            res = sign | ((exp + 112u) << 23) | (mant << 13);
        }
        else
        {
            res = sign;
        }
    }
    else if (exp == 31)
    {
        res = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        res = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &res, 4);
    return f;
}

/* Repack n_blocks of GGML Q6_K (210B/block) into 288B/block pre-unpacked AVX2 layout. */
void asi_v2_repack_q6k(const uint8_t *src, uint8_t *dst, size_t n_blocks)
{
    for (size_t b = 0; b < n_blocks; b++)
    {
        const uint8_t *block = src + b * 210;
        uint8_t *out = dst + b * 288;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = repack_fp16_to_fp32(d_raw);

        uint8_t *u6 = out;
        for (int half = 0; half < 2; half++)
        {
            for (int l = 0; l < 32; l++)
            {
                u6[l +  0] = (uint8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4));
                u6[l + 32] = (uint8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                u6[l + 64] = (uint8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4));
                u6[l + 96] = (uint8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4));
            }
            u6 += 128;
            ql += 64;
            qh += 32;
        }
        memcpy(out + 256, sc, 16);
        memcpy(out + 272, &d, 4);
        memset(out + 276, 0, 12);
    }
}
