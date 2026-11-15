# Lila Inference Engine — Build System
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)
CC := gcc
CFLAGS := -O3 -march=native -Wall -Wextra -std=c11 -pthread -fopenmp
LDFLAGS := -lm -lpthread -fopenmp
ifeq ($(UNAME_S),Linux)
    LDFLAGS += -Wl,-z,noexecstack
endif

ifeq ($(UNAME_M),x86_64)
    ASM := nasm
    ASMFLAGS := -f elf64
    ARCH_DIR := x86_64
    CFLAGS += -mavx2 -mfma -mf16c
else ifneq ($(filter $(UNAME_M),aarch64 arm64),)
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
ASI_SRC := asi/asi_runtime.c asi/asi_cli.c asi/lilavm.c asi/asi_v2.c asi/asi_arch.c asi/lila_asi.c
ASI_OBJ := $(ASI_SRC:.c=.o)

.PHONY: all clean test test-v2 lila-asi liblila-asi bench-roofline claim-check

all: lila-engine

lila-engine: $(KERN_OBJ) $(RT_OBJ) $(CLI_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "\n✅ Built lila-engine for $(UNAME_M)"
	@echo "   Run: ./lila-engine model.lila"

lila-asi: $(KERN_OBJ) $(RT_OBJ) $(ASI_OBJ) asi/libasi_v2.so asi/liblila_asi.so
	$(CC) $(CFLAGS) -o $@ $(KERN_OBJ) $(RT_OBJ) $(ASI_OBJ) $(LDFLAGS)
	@echo "\n✅ Built lila-asi for $(UNAME_M)"

liblila-asi: asi/liblila_asi.so asi/libasi_v2.so

asi/liblila_asi.so: asi/asi_v2.c asi/asi_arch.c asi/lila_asi.c asi/asi_v2.h asi/asi_arch.h asi/lila_asi.h runtime/q4k.h
	$(CC) $(CFLAGS) -fPIC -shared -D_GNU_SOURCE -o $@ asi/asi_v2.c asi/asi_arch.c asi/lila_asi.c -I runtime/ -I asi/ $(LDFLAGS)
	@echo "\n✅ Built asi/liblila_asi.so for $(UNAME_M)"

asi/libasi_v2.so: asi/asi_v2.c asi/asi_v2.h
	$(CC) $(CFLAGS) -fPIC -shared -D_GNU_SOURCE -o $@ asi/asi_v2.c -I asi/

asi/%.o: asi/%.c asi/*.h runtime/q4k.h
	$(CC) $(CFLAGS) -D_GNU_SOURCE -c -o $@ $< -I runtime/ -I asi/

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

test-v2: asi/asi_v2.o asi/asi_arch.o asi/lila_asi.o tests/test_asi_v2.c
	python3 tests/test_asi_v2.py /tmp/test_v2.asi
	$(CC) $(CFLAGS) -D_GNU_SOURCE -I runtime/ -I asi/ tests/test_asi_v2.c asi/asi_v2.o asi/asi_arch.o asi/lila_asi.o -o /tmp/test_asi_v2 $(LDFLAGS)
	/tmp/test_asi_v2 /tmp/test_v2.asi

tests/bench_roofline: tests/bench_roofline.c asi/asi_v2.o asi/asi_arch.o asi/lila_asi.o asi/asi_v2.h asi/asi_arch.h asi/lila_asi.h asi/asi_pool.h runtime/q4k.h
	$(CC) $(CFLAGS) -D_GNU_SOURCE -I runtime/ -I asi/ tests/bench_roofline.c asi/asi_v2.o asi/asi_arch.o asi/lila_asi.o -o $@ $(LDFLAGS)

bench-roofline: tests/bench_roofline
	./tests/bench_roofline $(ARGS)

claim-check:
	@if [ -x /home/ticket/workspaces/lila/environment/.venv/bin/python ]; then \
		/home/ticket/workspaces/lila/environment/.venv/bin/python tests/claim_check.py $(ARGS); \
	else \
		python3 tests/claim_check.py $(ARGS); \
	fi

clean:
	rm -f lila-engine lila-asi asi/liblila_asi.so asi/libasi_v2.so tests/bench_roofline $(KERN_OBJ) $(RT_OBJ) $(CLI_OBJ) $(ASI_OBJ)
