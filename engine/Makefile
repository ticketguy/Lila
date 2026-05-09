# Lila Inference Engine — Build System
# Builds both the legacy lila-engine AND the new lila-asi (Active System Image)
UNAME_M := $(shell uname -m)
CC := gcc
CFLAGS := -D_GNU_SOURCE -O3 -march=native -Wall -Wextra -std=c11 -pthread
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

# ═══════════════════════════════════════════════════════════════
#  Sources
# ═══════════════════════════════════════════════════════════════

# Assembly kernels (native, arch-specific)
KERN_SRC := $(wildcard kernels/$(ARCH_DIR)/*.S)
KERN_OBJ := $(KERN_SRC:.S=.o)

# Runtime (shared between both targets)
RT_SRC := runtime/model.c runtime/inference.c runtime/attention.c \
           runtime/transformer.c runtime/tokenizer.c runtime/detect.c \
           runtime/dispatch.c
RT_OBJ := $(RT_SRC:.c=.o)

# Legacy CLI (uses .lila format)
CLI_SRC := interface/cli.c
CLI_OBJ := $(CLI_SRC:.c=.o)

# ASI system (new — uses .asi format)
ASI_SRC := asi/asi_runtime.c asi/lilavm.c asi/asi_cli.c
ASI_OBJ := $(ASI_SRC:.c=.o)

# ═══════════════════════════════════════════════════════════════
#  Targets
# ═══════════════════════════════════════════════════════════════

.PHONY: all clean test asi legacy

all: lila-engine lila-asi

legacy: lila-engine
asi: lila-asi

# Legacy engine (loads .lila files)
lila-engine: $(KERN_OBJ) $(RT_OBJ) $(CLI_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "\n✅ Built lila-engine for $(UNAME_M)"
	@echo "   Run: ./lila-engine model.lila"

# ASI engine (loads .asi files — the new unified format)
lila-asi: $(KERN_OBJ) $(RT_OBJ) $(ASI_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "\n✅ Built lila-asi for $(UNAME_M)"
	@echo "   Run: ./lila-asi lila.asi"
	@echo "   One file. Everything. She thinks."

# ═══════════════════════════════════════════════════════════════
#  Compilation rules
# ═══════════════════════════════════════════════════════════════

kernels/$(ARCH_DIR)/%.o: kernels/$(ARCH_DIR)/%.S
ifeq ($(UNAME_M),x86_64)
	$(ASM) $(ASMFLAGS) -o $@ $<
else
	$(ASM) $(ASMFLAGS) -o $@ $<
endif

runtime/%.o: runtime/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I runtime/ -I asi/

interface/%.o: interface/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I runtime/

asi/%.o: asi/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I runtime/ -I asi/

# ═══════════════════════════════════════════════════════════════
#  Testing & Utilities
# ═══════════════════════════════════════════════════════════════

test: lila-engine lila-asi
	./lila-engine --test
	@echo "Legacy engine: OK"
	@echo ""

# Pack a model into .asi format
pack:
	@echo "Pack a model into .asi:"
	@echo "  python asi/pack_asi.py --model google/gemma-3-4b-it --output lila.asi"

# Inspect an .asi file
inspect:
	@echo "Usage: ./lila-asi <file.asi> --info"

clean:
	rm -f lila-engine lila-asi $(KERN_OBJ) $(RT_OBJ) $(CLI_OBJ) $(ASI_OBJ)
