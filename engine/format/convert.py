#!/usr/bin/env python3
"""
Convert a HuggingFace model (safetensors) to Lila's custom binary format.

Uses FigQuant from Little Fig for INT4 quantization.

Usage:
    python convert.py --model google/gemma-3-4b-it --output model.lila
"""

import argparse
import struct
import sys
import os

LILA_MAGIC = 0x4C494C41  # "LILA"
LILA_VERSION = 1


def convert(model_path: str, output_path: str, group_size: int = 128):
    """Convert HF model to Lila binary format."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"Loading model: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    
    print(f"Model config: layers={config.num_hidden_layers}, "
          f"hidden={config.hidden_size}, vocab={config.vocab_size}")
    
    # Try to import FigQuant for INT4
    try:
        sys.path.insert(0, os.path.expanduser("~/littlefig/src"))
        from little_fig.engine.figquant import figquant_quantize
        has_figquant = True
        print("Using FigQuant for INT4 quantization")
    except ImportError:
        has_figquant = False
        print("WARNING: FigQuant not available. Storing FP32 (large file).")
    
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("I", LILA_MAGIC))
        f.write(struct.pack("I", LILA_VERSION))
        f.write(struct.pack("I", config.num_hidden_layers))
        f.write(struct.pack("I", config.hidden_size))
        f.write(struct.pack("I", config.intermediate_size))
        f.write(struct.pack("I", config.num_attention_heads))
        f.write(struct.pack("I", getattr(config, "num_key_value_heads", config.num_attention_heads)))
        f.write(struct.pack("I", config.vocab_size))
        f.write(struct.pack("I", getattr(config, "max_position_embeddings", 4096)))
        
        # TODO: Write quantized weight tensors
        # For each linear layer: quantize with FigQuant, write codebook + indices + scales
        
        print(f"Header written. Full weight conversion TODO.")
        print(f"Output: {output_path}")
    
    del model
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="model.lila")
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    convert(args.model, args.output, args.group_size)
