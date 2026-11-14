#!/usr/bin/env python3
"""
Phase 1 ASI v2 round-trip generator for tests/test_asi_v2.c.
"""

import json
import sys
from pathlib import Path

ASI_DIR = Path(__file__).resolve().parent.parent / "asi"
sys.path.insert(0, str(ASI_DIR))

from asi_v2 import (
    ARCH_IDS,
    SEC_CHAT_TEMPLATE,
    SEC_METADATA,
    SEC_MODEL_CONFIG,
    SEC_TOKENIZER,
    SEC_WEIGHTS,
    TENSOR_LAYOUT_ROW_MAJOR,
    pack_model_config,
    pack_weights_section,
    write_asi_v2,
)


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/test_v2.asi")

    cfg = pack_model_config(
        arch_id=ARCH_IDS["hybrid_qwen35"],
        n_layers=24,
        hidden_size=1024,
        intermediate_size=3584,
        n_heads=8,
        n_kv_heads=2,
        head_dim=256,
        vocab_size=248320,
        max_seq_len=32768,
        rope_dim=64,
        rope_theta=10000000.0,
        rms_norm_eps=1e-6,
        embed_scale=1.0,
        full_attention_interval=4,
        ssm_conv_kernel=4,
        ssm_state_size=128,
        ssm_group_count=16,
        ssm_inner_size=2048,
        ssm_time_step_rank=16,
        default_quant_type=4,
    )

    weights = pack_weights_section(
        [
            ("blk.0.attn_q.weight", 4, TENSOR_LAYOUT_ROW_MAJOR, (1024, 1024), b"\x55" * 1024),
            ("token_embd.weight", 4, TENSOR_LAYOUT_ROW_MAJOR, (248320, 1024), b"\xAA" * 512),
        ]
    )

    tok_bytes = json.dumps({"bos": "<|endoftext|>", "eos": "<|im_end|>"}).encode("utf-8")
    tmpl_bytes = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}".encode("utf-8")
    meta_bytes = json.dumps({"model": "Qwen3.5-0.8B-test", "format": "asi-v2"}).encode("utf-8")

    write_asi_v2(
        out_path,
        arch_id=ARCH_IDS["hybrid_qwen35"],
        flags=0,
        sections=[
            (SEC_MODEL_CONFIG, 0, cfg),
            (SEC_WEIGHTS, 0, weights),
            (SEC_TOKENIZER, 0, tok_bytes),
            (SEC_CHAT_TEMPLATE, 0, tmpl_bytes),
            (SEC_METADATA, 0, meta_bytes),
        ],
    )
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
