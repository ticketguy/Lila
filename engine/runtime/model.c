#define _GNU_SOURCE
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/*
 * Load model weights via mmap — zero copy from disk.
 * The file is memory-mapped directly, so the OS handles
 * paging weights in/out as needed. Perfect for edge devices
 * with limited RAM.
 */

LilaModel *lila_load_model(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open model: %s\n", path);
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    
    void *mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap model\n");
        return NULL;
    }
    
    /* Advise the kernel we'll read sequentially during inference */
    madvise(mapped, file_size, MADV_SEQUENTIAL);
    
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
        munmap(model->mmap_addr, model->mmap_size);
    }
    /* Free KV cache */
    free(model->kv_cache.key_cache);
    free(model->kv_cache.value_cache);
    free(model);
}
