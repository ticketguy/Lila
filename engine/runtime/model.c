#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  PLATFORM ABSTRACTION                                                     */
/* ═══════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#include <windows.h>

static void *model_mmap(const char *path, size_t *out_size) {
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    *out_size = (size_t)li.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { CloseHandle(hFile); return NULL; }

    void *mapped = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);
    return mapped;
}

static void model_munmap(void *addr, size_t size) {
    (void)size;
    if (addr) UnmapViewOfFile(addr);
}

#else
/* POSIX */
#define _GNU_SOURCE
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static void *model_mmap(const char *path, size_t *out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    fstat(fd, &st);
    *out_size = st.st_size;

    void *mapped = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped == MAP_FAILED) return NULL;
    madvise(mapped, *out_size, MADV_SEQUENTIAL);
    return mapped;
}

static void model_munmap(void *addr, size_t size) {
    if (addr) munmap(addr, size);
}
#endif

/*
 * Load model weights via mmap — zero copy from disk.
 * The file is memory-mapped directly, so the OS handles
 * paging weights in/out as needed. Perfect for edge devices
 * with limited RAM.
 */

LilaModel *lila_load_model(const char *path) {
    size_t file_size = 0;
    void *mapped = model_mmap(path, &file_size);

    if (!mapped) {
        fprintf(stderr, "Failed to open/map model: %s\n", path);
        return NULL;
    }
    
    LilaModel *model = calloc(1, sizeof(LilaModel));
    model->mmap_addr = mapped;
    model->mmap_size = file_size;
    
    /* Parse header */
    uint8_t *ptr = (uint8_t *)mapped;
    memcpy(&model->magic, ptr, 4); ptr += 4;
    
    if (model->magic != LILA_MAGIC) {
        fprintf(stderr, "Invalid model magic: 0x%08X\n", model->magic);
        lila_free_model(model);
        return NULL;
    }
    
    memcpy(&model->version, ptr, 4); ptr += 4;
    
    /* Read config */
    memcpy(&model->n_layers, ptr, 4); ptr += 4;
    memcpy(&model->hidden_size, ptr, 4); ptr += 4;
    memcpy(&model->intermediate_size, ptr, 4); ptr += 4;
    memcpy(&model->n_heads, ptr, 4); ptr += 4;
    memcpy(&model->n_kv_heads, ptr, 4); ptr += 4;
    memcpy(&model->vocab_size, ptr, 4); ptr += 4;
    memcpy(&model->max_seq_len, ptr, 4); ptr += 4;
    
    model->head_dim = model->hidden_size / model->n_heads;
    model->rope_theta = 10000.0f;
    model->rms_norm_eps = 1e-6f;
    
    /* TODO: Parse weight tensors from mmap'd region */
    /* For now, this is the structural foundation */
    
    fprintf(stderr, "Loaded model: %d layers, hidden=%d, vocab=%d\n",
            model->n_layers, model->hidden_size, model->vocab_size);
    
    return model;
}

void lila_free_model(LilaModel *model) {
    if (!model) return;
    if (model->mmap_addr) {
        model_munmap(model->mmap_addr, model->mmap_size);
    }
    /* Free KV cache */
    free(model->kv_cache.key_cache);
    free(model->kv_cache.value_cache);
    free(model);
}
