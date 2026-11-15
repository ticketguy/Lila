#include "asi_v2.h"
#include "lila_asi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char     name[40];
    uint32_t quant_type;
    uint64_t cols;
    uint64_t rows;
    uint64_t data_offset;
    uint64_t data_size;
} SynthTensorSpec;

static uint64_t align_up_u64(uint64_t v, uint64_t align)
{
    return (v + align - 1u) & ~(align - 1u);
}

static void fill_f32_blob(uint8_t *dst, size_t n_floats, float base, float amp)
{
    float *f = (float *)dst;
    for (size_t i = 0; i < n_floats; ++i)
    {
        float phase = (float)((int)(i % 17u) - 8) / 8.0f;
        f[i] = base + amp * phase;
    }
}

static void fill_q4k_blob(uint8_t *dst, size_t n_blocks, uint32_t seed)
{
    for (size_t b = 0; b < n_blocks; ++b)
    {
        uint8_t *blk = dst + b * 144u;
        uint16_t d = 0x2800u;     /* 0.03125f */
        uint16_t dmin = 0x2000u;  /* 0.0078125f */
        memcpy(blk + 0, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (size_t k = 0; k < 12; ++k)
        {
            uint32_t v = (uint32_t)(b * 131u + k * 17u + seed);
            blk[4 + k] = (uint8_t)((v & 0x3Fu) | ((k >= 8) ? ((v >> 2) & 0xC0u) : 0u));
        }
        for (size_t k = 0; k < 128; ++k)
        {
            uint32_t v = (uint32_t)(b * 73u + k * 29u + seed);
            blk[16 + k] = (uint8_t)(v & 0xFFu);
        }
    }
}

static void fill_q6k_blob(uint8_t *dst, size_t n_blocks, uint32_t seed)
{
    for (size_t b = 0; b < n_blocks; ++b)
    {
        uint8_t *blk = dst + b * 210u;
        for (size_t k = 0; k < 128; ++k)
        {
            blk[k] = (uint8_t)((b * 41u + k * 13u + seed) & 0xFFu);
        }
        for (size_t k = 0; k < 64; ++k)
        {
            blk[128 + k] = (uint8_t)((b * 19u + k * 7u + seed) & 0xFFu);
        }
        int8_t *sc = (int8_t *)(blk + 192);
        for (size_t k = 0; k < 16; ++k)
        {
            sc[k] = (int8_t)(((int)((b + k + seed) % 13u)) - 6);
        }
        uint16_t d = 0x2800u; /* 0.03125f */
        memcpy(blk + 208, &d, 2);
    }
}

static size_t synth_tensor_bytes(uint32_t qtype, uint64_t cols, uint64_t rows)
{
    if (qtype == 0u) return (size_t)(cols * rows * sizeof(float));
    if (qtype == 12u) return (size_t)((cols / 256u) * rows * 144u);
    if (qtype == 14u) return (size_t)((cols / 256u) * rows * 210u);
    return 0;
}

static int add_synth_tensor(SynthTensorSpec *specs, size_t *count, size_t cap,
                            const char *name, uint32_t qtype, uint64_t cols, uint64_t rows)
{
    if (*count >= cap) return -1;
    SynthTensorSpec *s = &specs[*count];
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->quant_type = qtype;
    s->cols = cols;
    s->rows = rows;
    s->data_size = (uint64_t)synth_tensor_bytes(qtype, cols, rows);
    (*count)++;
    return 0;
}

static int write_synth_qwen35_asi(const char *path, uint8_t id_salt)
{
    /* 2-layer valid hybrid_qwen35 cartridge: blk.0 = DeltaNet, blk.1 = FullAttn */
    const uint32_t H = 256;
    const uint32_t I = 256;
    const uint32_t n_heads = 8;
    const uint32_t n_kv = 2;
    const uint32_t hd = 32;
    const uint32_t vocab = 64;
    const uint32_t ssm_conv = 4;
    const uint32_t ssm_state = 128;
    const uint32_t ssm_groups = 2;
    const uint32_t ssm_inner = 256;
    const uint32_t ssm_r = 2;
    const uint32_t qkv_dim = ssm_inner * 3u;

    SynthTensorSpec specs[32];
    size_t n_t = 0;

    if (add_synth_tensor(specs, &n_t, 32, "token_embd.weight", 14, H, vocab) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "output_norm.weight", 0, H, 1) != 0 ||
        /* Layer 0: DeltaNet */
        add_synth_tensor(specs, &n_t, 32, "blk.0.attn_norm.weight", 0, H, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.post_attention_norm.weight", 0, H, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ffn_gate.weight", 12, H, I) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ffn_up.weight", 12, H, I) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ffn_down.weight", 12, I, H) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.attn_qkv.weight", 12, H, qkv_dim) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.attn_gate.weight", 12, H, ssm_inner) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_a", 0, ssm_r, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_alpha.weight", 12, H, ssm_r) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_beta.weight", 12, H, ssm_r) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_conv1d.weight", 0, ssm_conv, qkv_dim) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_dt.bias", 0, ssm_r, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_norm.weight", 0, ssm_inner / ssm_groups, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.0.ssm_out.weight", 12, ssm_inner, H) != 0 ||
        /* Layer 1: FullAttn */
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_norm.weight", 0, H, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.post_attention_norm.weight", 0, H, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.ffn_gate.weight", 12, H, I) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.ffn_up.weight", 12, H, I) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.ffn_down.weight", 12, I, H) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_q.weight", 12, H, n_heads * 2u * hd) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_k.weight", 12, H, n_kv * hd) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_v.weight", 12, H, n_kv * hd) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_q_norm.weight", 0, hd, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_k_norm.weight", 0, hd, 1) != 0 ||
        add_synth_tensor(specs, &n_t, 32, "blk.1.attn_output.weight", 12, n_heads * hd, H) != 0)
    {
        return -1;
    }

    uint64_t w_hdr_bytes = sizeof(AsiV2WeightsHeader) + n_t * sizeof(AsiV2TensorEntry);
    uint64_t w_data_start = align_up_u64(w_hdr_bytes, 64u);
    uint64_t cur_rel = 0;
    for (size_t i = 0; i < n_t; ++i)
    {
        specs[i].data_offset = cur_rel;
        cur_rel = align_up_u64(cur_rel + specs[i].data_size, 64u);
    }

    uint64_t cfg_off = ASI_V2_PAGE_SIZE;
    uint64_t w_off = ASI_V2_PAGE_SIZE * 2u;
    uint64_t w_sec_size = w_data_start + cur_rel;
    uint64_t tok_off = align_up_u64(w_off + w_sec_size, ASI_V2_PAGE_SIZE);
    const char *tok_json = "{}";
    uint64_t tok_size = 2u;
    uint64_t total_size = align_up_u64(tok_off + tok_size, ASI_V2_PAGE_SIZE);

    uint8_t *buf = (uint8_t *)calloc(1, (size_t)total_size);
    if (!buf) return -1;

    AsiV2ModelConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.arch_id = ASI_ARCH_HYBRID_QWEN35;
    cfg.n_layers = 2;
    cfg.hidden_size = H;
    cfg.intermediate_size = I;
    cfg.n_heads = n_heads;
    cfg.n_kv_heads = n_kv;
    cfg.head_dim = hd;
    cfg.vocab_size = vocab;
    cfg.max_seq_len = 128;
    cfg.rope_dim = 16;
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-6f;
    cfg.embed_scale = 1.0f;
    cfg.full_attention_interval = 2;
    cfg.ssm_conv_kernel = ssm_conv;
    cfg.ssm_state_size = ssm_state;
    cfg.ssm_group_count = ssm_groups;
    cfg.ssm_inner_size = ssm_inner;
    cfg.ssm_time_step_rank = ssm_r;
    cfg.default_quant_type = ASI_QUANT_Q4_K;
    memcpy(buf + cfg_off, &cfg, sizeof(cfg));

    AsiV2WeightsHeader wh;
    wh.n_tensors = (uint32_t)n_t;
    wh.flags = 0;
    wh.data_offset = w_data_start;
    memcpy(buf + w_off, &wh, sizeof(wh));

    uint8_t *w_tbl_dst = buf + w_off + sizeof(wh);
    uint8_t *w_blob = buf + w_off + w_data_start;
    for (size_t i = 0; i < n_t; ++i)
    {
        AsiV2TensorEntry te;
        memset(&te, 0, sizeof(te));
        memcpy(te.name, specs[i].name, sizeof(te.name));
        te.quant_type = specs[i].quant_type;
        te.layout_flags = ASI_TENSOR_LAYOUT_ROW_MAJOR;
        te.shape[0] = specs[i].cols;
        te.shape[1] = specs[i].rows;
        te.data_offset = specs[i].data_offset;
        te.data_size = specs[i].data_size;
        memcpy(w_tbl_dst + i * sizeof(te), &te, sizeof(te));

        uint8_t *dst = w_blob + specs[i].data_offset;
        if (specs[i].quant_type == 0u)
        {
            size_t nf = (size_t)(specs[i].cols * specs[i].rows);
            if (strstr(specs[i].name, "ssm_a") != NULL)
                fill_f32_blob(dst, nf, -0.5f, 0.05f);
            else if (strstr(specs[i].name, "conv1d") != NULL)
                fill_f32_blob(dst, nf, 0.05f, 0.02f);
            else if (strstr(specs[i].name, "dt.bias") != NULL)
                fill_f32_blob(dst, nf, 0.1f, 0.02f);
            else
                fill_f32_blob(dst, nf, 1.0f, 0.05f);
        }
        else if (specs[i].quant_type == 12u)
        {
            fill_q4k_blob(dst, (size_t)specs[i].data_size / 144u, (uint32_t)(i + 1u));
        }
        else if (specs[i].quant_type == 14u)
        {
            fill_q6k_blob(dst, (size_t)specs[i].data_size / 210u, (uint32_t)(i + 101u));
        }
    }

    memcpy(buf + tok_off, tok_json, (size_t)tok_size);

    AsiV2SectionEntry secs[3];
    memset(secs, 0, sizeof(secs));
    secs[0].type = ASI_SEC_MODEL_CONFIG;
    secs[0].offset = cfg_off;
    secs[0].size = sizeof(cfg);
    secs[0].checksum = asi_v2_crc64(buf + cfg_off, sizeof(cfg));

    secs[1].type = ASI_SEC_WEIGHTS;
    secs[1].offset = w_off;
    secs[1].size = w_sec_size;
    secs[1].checksum = asi_v2_crc64(buf + w_off, (size_t)w_sec_size);

    secs[2].type = ASI_SEC_TOKENIZER;
    secs[2].offset = tok_off;
    secs[2].size = tok_size;
    secs[2].checksum = asi_v2_crc64(buf + tok_off, (size_t)tok_size);
    memcpy(buf + sizeof(AsiV2Header), secs, sizeof(secs));

    AsiV2Header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = ASI_V2_MAGIC;
    hdr.version = ASI_V2_VERSION;
    hdr.arch_id = ASI_ARCH_HYBRID_QWEN35;
    hdr.n_sections = 3;
    hdr.total_size = total_size;
    hdr.section_table_offset = sizeof(AsiV2Header);
    for (size_t i = 0; i < 24; ++i)
    {
        hdr.identity_hash[i] = (uint8_t)(0xA0u + i + id_salt);
    }
    memcpy(buf, &hdr, sizeof(hdr));

    FILE *fp = fopen(path, "wb");
    int rc = -1;
    if (fp)
    {
        if (fwrite(buf, 1, (size_t)total_size, fp) == (size_t)total_size) rc = 0;
        fclose(fp);
    }
    free(buf);
    return rc;
}

static int corrupt_file_copy(const char *src_path, const char *dst_path,
                             size_t flip_offset, int truncate_tail_bytes, int append_byte)
{
    FILE *fi = fopen(src_path, "rb");
    if (!fi) return -1;
    if (fseek(fi, 0, SEEK_END) != 0) { fclose(fi); return -1; }
    long sz_l = ftell(fi);
    if (sz_l <= 0) { fclose(fi); return -1; }
    size_t sz = (size_t)sz_l;
    rewind(fi);
    uint8_t *buf = (uint8_t *)malloc(sz + 1u);
    if (!buf) { fclose(fi); return -1; }
    if (fread(buf, 1, sz, fi) != sz) { free(buf); fclose(fi); return -1; }
    fclose(fi);

    if (flip_offset < sz) buf[flip_offset] ^= 0x5Au;
    size_t out_sz = sz;
    if (truncate_tail_bytes > 0 && (size_t)truncate_tail_bytes < sz)
        out_sz = sz - (size_t)truncate_tail_bytes;
    if (append_byte)
    {
        buf[out_sz] = 0xAAu;
        out_sz++;
    }

    FILE *fo = fopen(dst_path, "wb");
    if (!fo) { free(buf); return -1; }
    int rc = (fwrite(buf, 1, out_sz, fo) == out_sz) ? 0 : -1;
    fclose(fo);
    free(buf);
    return rc;
}

static float max_abs_diff(const float *a, const float *b, uint32_t n)
{
    float mx = 0.0f;
    for (uint32_t i = 0; i < n; ++i)
    {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static int run_phase1_check(const char *path)
{
    AsiV2Cartridge cart;
    char err[256] = {0};
    if (asi_v2_open(path, &cart, err, sizeof(err)) != 0)
    {
        fprintf(stderr, "FAIL: asi_v2_open failed: %s\n", err);
        return 1;
    }

    printf("ASI v%u Cartridge OK: arch=%s layers=%u hidden=%u tensors=%u\n",
           cart.header.version,
           asi_v2_arch_name(cart.header.arch_id),
           cart.config.n_layers,
           cart.config.hidden_size,
           cart.n_tensors);

    const AsiV2TensorEntry *t = asi_v2_find_tensor(&cart, "blk.0.attn_q.weight");
    if (!t || t->data_size == 0)
    {
        fprintf(stderr, "FAIL: expected tensor blk.0.attn_q.weight not found\n");
        asi_v2_close(&cart);
        return 1;
    }

    if (!asi_v2_find_section(&cart, ASI_SEC_CHAT_TEMPLATE))
    {
        fprintf(stderr, "FAIL: CHAT_TEMPLATE section missing\n");
        asi_v2_close(&cart);
        return 1;
    }

    /* Negative test: corrupt 1 byte in memory and verify CRC-64 rejects it */
    uint8_t *copy = (uint8_t *)malloc(cart.file_size);
    memcpy(copy, cart.base, cart.file_size);
    const AsiV2SectionEntry *cfg = asi_v2_find_section(&cart, ASI_SEC_MODEL_CONFIG);
    copy[cfg->offset] ^= 0xFF;

    AsiV2Cartridge bad;
    if (asi_v2_validate_memory(copy, cart.file_size, &bad, err, sizeof(err)) == 0)
    {
        fprintf(stderr, "FAIL: corrupted section was not rejected\n");
        free(copy);
        asi_v2_close(&cart);
        return 1;
    }
    printf("PASS: corrupted cartridge rejected (%s)\n", err);

    free(copy);
    asi_v2_close(&cart);
    printf("PASS: ASI v2 spec & C validator verified\n");
    return 0;
}

static int run_session_and_prefix_tests(const char *model_path, const char *other_model_path)
{
    char err[256] = {0};
    LilaAsiModel *model = lila_asi_open(model_path, err, sizeof(err));
    if (!model)
    {
        fprintf(stderr, "FAIL: lila_asi_open(%s) failed: %s\n", model_path, err);
        return 1;
    }

    AsiV2Header hdr;
    AsiV2ModelConfig cfg;
    if (lila_asi_model_header(model, &hdr) != 0 || lila_asi_model_info(model, &cfg) != 0)
    {
        fprintf(stderr, "FAIL: lila_asi_model_header/info failed\n");
        lila_asi_close(model);
        return 1;
    }
    if (hdr.magic != ASI_V2_MAGIC || hdr.version != ASI_V2_VERSION ||
        hdr.arch_id != ASI_ARCH_HYBRID_QWEN35)
    {
        fprintf(stderr, "FAIL: unexpected header fields in %s\n", model_path);
        lila_asi_close(model);
        return 1;
    }

    uint32_t vocab = cfg.vocab_size;
    float *logits_tok = (float *)calloc(vocab, sizeof(float));
    float *logits_bat = (float *)calloc(vocab, sizeof(float));
    float *logits_1t  = (float *)calloc(vocab, sizeof(float));
    float *logits_rel = (float *)calloc(vocab, sizeof(float));
    if (!logits_tok || !logits_bat || !logits_1t || !logits_rel)
    {
        fprintf(stderr, "FAIL: OOM allocating logits buffers\n");
        free(logits_tok); free(logits_bat); free(logits_1t); free(logits_rel);
        lila_asi_close(model);
        return 1;
    }

    LilaAsiSession *s_tok = lila_asi_session_create(model, 64, 2, err, sizeof(err));
    LilaAsiSession *s_bat = lila_asi_session_create(model, 64, 2, err, sizeof(err));
    LilaAsiSession *s_1t  = lila_asi_session_create(model, 64, 1, err, sizeof(err));
    LilaAsiSession *s_rel = lila_asi_session_create(model, 64, 2, err, sizeof(err));
    if (!s_tok || !s_bat || !s_1t || !s_rel)
    {
        fprintf(stderr, "FAIL: lila_asi_session_create failed: %s\n", err);
        lila_asi_session_free(s_tok); lila_asi_session_free(s_bat);
        lila_asi_session_free(s_1t);  lila_asi_session_free(s_rel);
        free(logits_tok); free(logits_bat); free(logits_1t); free(logits_rel);
        lila_asi_close(model);
        return 1;
    }

    const int32_t prefix[4] = {1, 5 % (int32_t)vocab, 17 % (int32_t)vocab, 9 % (int32_t)vocab};
    for (uint32_t p = 0; p < 4; ++p)
    {
        if (lila_asi_eval_token(s_tok, prefix[p], p, logits_tok) != 0 ||
            lila_asi_eval_token(s_1t,  prefix[p], p, logits_1t) != 0)
        {
            fprintf(stderr, "FAIL: eval_token failed during prefix\n");
            goto fail_sessions;
        }
    }
    if (lila_asi_eval_batch(s_bat, prefix, 4, 0, logits_bat) != 0)
    {
        fprintf(stderr, "FAIL: eval_batch failed\n");
        goto fail_sessions;
    }

    float diff_bat_tok = max_abs_diff(logits_bat, logits_tok, vocab);
    float diff_1t_2t   = max_abs_diff(logits_1t, logits_tok, vocab);
    printf("Parity check (%s): batch_vs_token=%.6f, 1T_vs_2T=%.6f\n",
           model_path, diff_bat_tok, diff_1t_2t);
    if (diff_bat_tok != 0.0f || diff_1t_2t != 0.0f)
    {
        fprintf(stderr, "FAIL: expected exact 0.000000 parity (bat_tok=%.8g 1t_2t=%.8g)\n",
                diff_bat_tok, diff_1t_2t);
        goto fail_sessions;
    }

    /* Continue 2 tokens after prefix on s_bat */
    const int32_t cont0 = 23 % (int32_t)vocab;
    const int32_t cont1 = 7 % (int32_t)vocab;
    if (lila_asi_session_save_prefix(s_bat, 4) != 0)
    {
        fprintf(stderr, "FAIL: lila_asi_session_save_prefix failed\n");
        goto fail_sessions;
    }

    const char *pfx_path = "/tmp/test_asi_v2_prefix.bin";
    const char *bad_path = "/tmp/test_asi_v2_prefix_bad.bin";
    uint8_t key_hash[32];
    for (size_t i = 0; i < 32; ++i) key_hash[i] = (uint8_t)(0x31u + i * 3u);

    if (lila_asi_session_save_prefix_file(s_bat, key_hash, pfx_path) != 0)
    {
        fprintf(stderr, "FAIL: lila_asi_session_save_prefix_file failed\n");
        goto fail_sessions;
    }

    if (lila_asi_eval_token(s_bat, cont0, 4, logits_bat) != 0 ||
        lila_asi_eval_token(s_bat, cont1, 5, logits_bat) != 0)
    {
        fprintf(stderr, "FAIL: continuation eval_token failed on s_bat\n");
        goto fail_sessions;
    }

    /* Load saved disk snapshot into fresh session s_rel */
    uint32_t loaded_n = 0;
    if (lila_asi_session_load_prefix_file(s_rel, key_hash, pfx_path, &loaded_n) != 0 || loaded_n != 4)
    {
        fprintf(stderr, "FAIL: lila_asi_session_load_prefix_file failed (loaded_n=%u)\n", loaded_n);
        goto fail_sessions;
    }
    if (lila_asi_eval_token(s_rel, cont0, 4, logits_rel) != 0 ||
        lila_asi_eval_token(s_rel, cont1, 5, logits_rel) != 0)
    {
        fprintf(stderr, "FAIL: continuation eval_token failed on reloaded s_rel\n");
        goto fail_sessions;
    }

    float diff_rel = max_abs_diff(logits_bat, logits_rel, vocab);
    printf("Disk prefix round-trip diff=%.6f (memcmp=%d)\n",
           diff_rel, memcmp(logits_bat, logits_rel, (size_t)vocab * sizeof(float)));
    if (diff_rel != 0.0f)
    {
        fprintf(stderr, "FAIL: disk prefix snapshot reload mismatch (%.8g)\n", diff_rel);
        goto fail_sessions;
    }

    /* Negative tests on prefix snapshot file */
    uint8_t wrong_key[32];
    memcpy(wrong_key, key_hash, sizeof(wrong_key));
    wrong_key[7] ^= 0x01u;
    if (lila_asi_session_load_prefix_file(s_rel, wrong_key, pfx_path, &loaded_n) == 0)
    {
        fprintf(stderr, "FAIL: wrong key_hash was accepted\n");
        goto fail_sessions;
    }
    if (lila_asi_session_load_prefix_file(s_rel, key_hash, "/tmp/does_not_exist_asi_pfx.bin", &loaded_n) == 0)
    {
        fprintf(stderr, "FAIL: nonexistent prefix file was accepted\n");
        goto fail_sessions;
    }

    /* Corrupt header byte (fp32_ref at offset 72), magic byte (0), payload byte, truncated, trailing */
    size_t offsets[3] = {0u, 72u, 200u};
    for (int k = 0; k < 3; ++k)
    {
        if (corrupt_file_copy(pfx_path, bad_path, offsets[k], 0, 0) != 0) goto fail_sessions;
        if (lila_asi_session_load_prefix_file(s_rel, key_hash, bad_path, &loaded_n) == 0)
        {
            fprintf(stderr, "FAIL: tampered prefix byte at offset %zu was accepted\n", offsets[k]);
            unlink(bad_path);
            goto fail_sessions;
        }
    }
    if (corrupt_file_copy(pfx_path, bad_path, (size_t)-1, 8, 0) != 0 ||
        lila_asi_session_load_prefix_file(s_rel, key_hash, bad_path, &loaded_n) == 0)
    {
        fprintf(stderr, "FAIL: truncated prefix file was accepted\n");
        unlink(bad_path);
        goto fail_sessions;
    }
    if (corrupt_file_copy(pfx_path, bad_path, (size_t)-1, 0, 1) != 0 ||
        lila_asi_session_load_prefix_file(s_rel, key_hash, bad_path, &loaded_n) == 0)
    {
        fprintf(stderr, "FAIL: trailing-garbage prefix file was accepted\n");
        unlink(bad_path);
        goto fail_sessions;
    }
    unlink(bad_path);

    /* Ensure failed loads left s_rel's valid in-memory prefix intact */
    loaded_n = 0;
    if (lila_asi_session_restore_prefix(s_rel, &loaded_n) != 0 || loaded_n != 4 ||
        lila_asi_eval_token(s_rel, cont0, 4, logits_rel) != 0 ||
        lila_asi_eval_token(s_rel, cont1, 5, logits_rel) != 0 ||
        max_abs_diff(logits_bat, logits_rel, vocab) != 0.0f)
    {
        fprintf(stderr, "FAIL: failed prefix loads corrupted s_rel rollback state\n");
        goto fail_sessions;
    }

    if (other_model_path)
    {
        LilaAsiModel *other_m = lila_asi_open(other_model_path, err, sizeof(err));
        if (!other_m)
        {
            fprintf(stderr, "FAIL: could not open other model %s: %s\n", other_model_path, err);
            goto fail_sessions;
        }
        LilaAsiSession *s_other = lila_asi_session_create(other_m, 64, 1, err, sizeof(err));
        if (!s_other)
        {
            lila_asi_close(other_m);
            goto fail_sessions;
        }
        if (lila_asi_session_load_prefix_file(s_other, key_hash, pfx_path, &loaded_n) == 0)
        {
            fprintf(stderr, "FAIL: prefix file from different model identity_hash was accepted\n");
            lila_asi_session_free(s_other);
            lila_asi_close(other_m);
            goto fail_sessions;
        }
        lila_asi_session_free(s_other);
        lila_asi_close(other_m);
    }

    unlink(pfx_path);
    printf("PASS: Session parity + disk prefix snapshot round-trip & tamper checks passed!\n");

    lila_asi_session_free(s_tok); lila_asi_session_free(s_bat);
    lila_asi_session_free(s_1t);  lila_asi_session_free(s_rel);
    free(logits_tok); free(logits_bat); free(logits_1t); free(logits_rel);
    lila_asi_close(model);
    return 0;

fail_sessions:
    unlink("/tmp/test_asi_v2_prefix.bin");
    unlink("/tmp/test_asi_v2_prefix_bad.bin");
    lila_asi_session_free(s_tok); lila_asi_session_free(s_bat);
    lila_asi_session_free(s_1t);  lila_asi_session_free(s_rel);
    free(logits_tok); free(logits_bat); free(logits_1t); free(logits_rel);
    lila_asi_close(model);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: test_asi_v2 <cartridge.asi> [--session-test <qwen35.asi>]\n");
        return 2;
    }

    const char *phase1_path = NULL;
    const char *ext_session_path = NULL;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--session-test") == 0 && i + 1 < argc)
        {
            ext_session_path = argv[++i];
        }
        else if (!phase1_path)
        {
            phase1_path = argv[i];
        }
    }

    if (phase1_path && run_phase1_check(phase1_path) != 0)
    {
        return 1;
    }

    const char *synth_a = "/tmp/test_synth_q35_a.asi";
    const char *synth_b = "/tmp/test_synth_q35_b.asi";
    if (write_synth_qwen35_asi(synth_a, 0) != 0 || write_synth_qwen35_asi(synth_b, 1) != 0)
    {
        fprintf(stderr, "FAIL: could not write synthetic Qwen3.5 test cartridges\n");
        unlink(synth_a);
        unlink(synth_b);
        return 1;
    }

    int rc = run_session_and_prefix_tests(synth_a, synth_b);
    unlink(synth_a);
    unlink(synth_b);
    if (rc != 0) return rc;

    if (ext_session_path)
    {
        rc = run_session_and_prefix_tests(ext_session_path, NULL);
        if (rc != 0) return rc;
    }

    return 0;
}
