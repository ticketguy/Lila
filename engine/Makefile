# Lila Inference Engine — Build System
# Detects architecture, assembles kernels, links runtime

UNAME_M := $(shell uname -m)
CC := gcc
CFLAGS := -O3 -march=native -Wall -Wextra -std=c11 -pthread
LDFLAGS := -lm -lpthread

# Architecture detection
ifeq ($(UNAME_M),x86_64)
    ASM := nasm
    ASMFLAGS := -f elf64
    ARCH_DIR := x86_64
    CFLAGS += -mavx2 -mfma
    # Check for AVX-512
    HAS_AVX512 := $(shell grep -c avx512f /proc/cpuinfo 2>/dev/null || echo 0)
    ifneq ($(HAS_AVX512),0)
        CFLAGS += -mavx512f -mavx512bw -mavx512vl
    endif
else ifeq ($(UNAME_M),aarch64)
    ASM := as
    ASMFLAGS :=
    ARCH_DIR := arm64
else
    $(error Unsupported architecture: $(UNAME_M))
endif

# Source files
KERN_SRC := $(wildcard kernels/$(ARCH_DIR)/*.S)
KERN_OBJ := $(KERN_SRC:.S=.o)
RT_SRC := $(wildcard runtime/*.c)
RT_OBJ := $(RT_SRC:.c=.o)
IF_SRC := $(wildcard interface/*.c)
IF_OBJ := $(IF_SRC:.c=.o)

# Targets
.PHONY: all clean test bench

all: lila-engine

lila-engine: $(KERN_OBJ) $(RT_OBJ) $(IF_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built lila-engine for $(UNAME_M)"

# Assembly kernels
kernels/$(ARCH_DIR)/%.o: kernels/$(ARCH_DIR)/%.S
ifeq ($(UNAME_M),x86_64)
	$(ASM) $(ASMFLAGS) -o $@ $<
else
	$(ASM) $(ASMFLAGS) -o $@ $<
endif

# C runtime
runtime/%.o: runtime/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# C interface
interface/%.o: interface/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Tests
test: lila-engine
	./lila-engine --test

bench: lila-engine
	./lila-engine --bench

clean:
	rm -f lila-engine $(KERN_OBJ) $(RT_OBJ) $(IF_OBJ)
