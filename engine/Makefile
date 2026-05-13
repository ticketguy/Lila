# Lila Inference Engine — Build System
UNAME_M := $(shell uname -m)
CC := gcc
CFLAGS := -O3 -march=native -Wall -Wextra -std=c11 -pthread
LDFLAGS := -lm -lpthread

ifeq ($(UNAME_M),x86_64)
    ASM := nasm
    ASMFLAGS := -f elf64
    ARCH_DIR := x86_64
    CFLAGS += -mavx2 -mfma
else ifeq ($(UNAME_M),aarch64)
    ASM := as
    ASMFLAGS :=
    ARCH_DIR := arm64
endif

# Sources
KERN_SRC := $(wildcard kernels/$(ARCH_DIR)/*.S)
KERN_OBJ := $(KERN_SRC:.S=.o)
RT_SRC := runtime/model.c runtime/inference.c runtime/attention.c \
           runtime/transformer.c runtime/tokenizer.c runtime/detect.c \
           runtime/dispatch.c
RT_OBJ := $(RT_SRC:.c=.o)
CLI_SRC := interface/cli.c
CLI_OBJ := $(CLI_SRC:.c=.o)

.PHONY: all clean test

all: lila-engine

lila-engine: $(KERN_OBJ) $(RT_OBJ) $(CLI_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "\n✅ Built lila-engine for $(UNAME_M)"
	@echo "   Run: ./lila-engine model.lila"

kernels/$(ARCH_DIR)/%.o: kernels/$(ARCH_DIR)/%.S
ifeq ($(UNAME_M),x86_64)
	$(ASM) $(ASMFLAGS) -o $@ $<
else
	$(ASM) $(ASMFLAGS) -o $@ $<
endif

runtime/%.o: runtime/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I runtime/

interface/%.o: interface/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I runtime/

test: lila-engine
	./lila-engine --test

clean:
	rm -f lila-engine $(KERN_OBJ) $(RT_OBJ) $(CLI_OBJ)
