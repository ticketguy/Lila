#!/usr/bin/env python3
"""
Convert HuggingFace model → Lila binary format (.lila)

Performs FigQuant INT4 quantization on all linear layers.
Output is directly mmap-loadable by the C engine.

File layout:
  [Header: 36 bytes]
  [Token Embedding: vocab_size * hidden_size * 4 bytes (FP32)]
  [Per-layer weights: quantized with FigQuant]
  [Final norm: hidden_size * 4 bytes (FP32)]
  [LM Head: vocab_size * hidden_size * 4 bytes (FP32)]

Usage:
    python convert.py --model google/gemma-3-4b-it --output model.lila
    python convert.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output tinyllama.lila
"""

import argparse
import struct
import sys
import os
import numpy as np

LILA_MAGIC = 0x4C494C41
LILA_VERSION = 1
GROUP_SIZE = 128


def quantize_int4(weight_np, group_size=128):
    """
    FigQuant-style INT4 quantization in numpy.
    Returns: (packed_indices, codebook, scales)
    """
    rows, cols = weight_np.shape
    flat = weight_np.reshape(-1).astype(np.float32)
    numel = flat.size
    
    # Pad to multiple of group_size
    pad = (group_size - numel % group_size) % group_size
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    
    grouped = flat.reshape(-1, group_size)
    n_groups = grouped.shape[0]
    
    # Per-group absmax scaling
    scales = np.abs(grouped).max(axis=1).clip(min=1e-10).astype(np.float32)
    scaled = grouped / scales[:, None]  # → [-1, 1]
    
    # NF4 codebook (initial)
    codebook = np.array([-1.0,-0.6962,-0.5251,-0.3949,-0.2844,-0.1848,-0.0911,0.0,
                          0.0796,0.1609,0.2461,0.3379,0.4407,0.5626,0.7230,1.0], dtype=np.float32)
    
    # K-means refinement (8 iterations)
    all_vals = scaled.reshape(-1)
    for _ in range(8):
        dists = np.abs(all_vals[:, None] - codebook[None, :])
        assignments = dists.argmin(axis=1)
        for i in range(16):
            mask = assignments == i
            if mask.sum() > 0:
                codebook[i] = all_vals[mask].mean()
    codebook[np.abs(codebook).argmin()] = 0.0
    
    # Final assignment
    all_scaled = scaled.reshape(-1)
    dists = np.abs(all_scaled[:, None] - codebook[None, :])
    indices = dists.argmin(axis=1).astype(np.uint8)
    
    # Pack 2 indices per byte
    indices_trimmed = indices[:numel + pad]
    packed = (indices_trimmed[0::2] | (indices_trimmed[1::2] << 4)).astype(np.uint8)
    
    return packed, codebook, scales


def write_quant_weight(f, weight_np, group_size=128):
    """Quantize and write a weight tensor to file."""
    rows, cols = weight_np.shape
    packed, codebook, scales = quantize_int4(weight_np, group_size)
    
    # Write metadata
    f.write(struct.pack("ii", rows, cols))
    # Write codebook (16 floats = 64 bytes)
    f.write(codebook.tobytes())
    # Write scales
    f.write(scales.tobytes())
    # Write packed indices
    f.write(packed.tobytes())
    
    return packed.nbytes + codebook.nbytes + scales.nbytes + 8


def write_fp32_tensor(f, tensor_np):
    """Write a tensor as raw FP32."""
    data = tensor_np.astype(np.float32).tobytes()
    f.write(data)
    return len(data)


def convert(model_path: str, output_path: str, group_size: int = 128):
    """Convert HF model to Lila format."""
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    
    print(f"Loading model: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    vocab_size = config.vocab_size
    max_seq = getattr(config, "max_position_embeddings", 4096)
    
    print(f"Config: {n_layers} layers, hidden={hidden}, inter={intermediate}, "
          f"heads={n_heads}, kv_heads={n_kv_heads}, vocab={vocab_size}")
    
    total_bytes = 0
    with open(output_path, "wb") as f:
        # ── Header (36 bytes) ──
        f.write(struct.pack("I", LILA_MAGIC))
        f.write(struct.pack("I", LILA_VERSION))
        f.write(struct.pack("I", n_layers))
        f.write(struct.pack("I", hidden))
        f.write(struct.pack("I", intermediate))
        f.write(struct.pack("I", n_heads))
        f.write(struct.pack("I", n_kv_heads))
        f.write(struct.pack("I", vocab_size))
        f.write(struct.pack("I", max_seq))
        total_bytes += 36
        print("  Header written")
        
        # ── Token Embedding (FP32) ──
        embed = model.get_input_embeddings().weight.data.numpy()
        total_bytes += write_fp32_tensor(f, embed)
        print(f"  Embedding: {embed.shape} ({embed.nbytes/1e6:.1f} MB)")
        
        # ── Transformer Layers ──
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx] if hasattr(model, 'model') else model.transformer.h[layer_idx]
            
            # Find weight tensors by common patterns
            layer_state = {k: v.data.numpy() for k, v in layer.named_parameters()}
            
            # Attention projections
            for proj_name in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                             "self_attn.v_proj.weight", "self_attn.o_proj.weight"]:
                if proj_name in layer_state:
                    total_bytes += write_quant_weight(f, layer_state[proj_name], group_size)
                else:
                    # Try alternate naming
                    alt = proj_name.replace("self_attn.", "attn.")
                    if alt in layer_state:
                        total_bytes += write_quant_weight(f, layer_state[alt], group_size)
                    else:
                        # Write zero placeholder
                        f.write(struct.pack("ii", 0, 0))
                        total_bytes += 8
            
            # MLP projections
            for proj_name in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
                if proj_name in layer_state:
                    total_bytes += write_quant_weight(f, layer_state[proj_name], group_size)
                else:
                    f.write(struct.pack("ii", 0, 0))
                    total_bytes += 8
            
            # Layer norms (FP32, small)
            for norm_name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                if norm_name in layer_state:
                    total_bytes += write_fp32_tensor(f, layer_state[norm_name])
                else:
                    total_bytes += write_fp32_tensor(f, np.ones(hidden, dtype=np.float32))
            
            if (layer_idx + 1) % 4 == 0:
                print(f"  Layer {layer_idx+1}/{n_layers} done")
        
        # ── Final Norm (FP32) ──
        final_norm = None
        for name, param in model.named_parameters():
            if "final" in name and "norm" in name and "weight" in name:
                final_norm = param.data.numpy()
                break
            elif name == "model.norm.weight":
                final_norm = param.data.numpy()
                break
        if final_norm is None:
            final_norm = np.ones(hidden, dtype=np.float32)
        total_bytes += write_fp32_tensor(f, final_norm)
        print(f"  Final norm written")
        
        # ── LM Head (FP32 — tied with embedding in many models) ──
        lm_head = model.get_output_embeddings()
        if lm_head is not None and lm_head.weight is not model.get_input_embeddings().weight:
            total_bytes += write_fp32_tensor(f, lm_head.weight.data.numpy())
            print(f"  LM Head written (separate)")
        else:
            # Tied weights — mark with special flag
            f.write(struct.pack("I", 0xFFFFFFFF))  # tied flag
            total_bytes += 4
            print(f"  LM Head: tied with embedding")
    
    # ── Export vocab ──
    vocab_path = output_path.replace(".lila", ".vocab")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(vocab_path, "w", encoding="utf-8") as vf:
            for i in range(min(vocab_size, len(tokenizer))):
                token = tokenizer.convert_ids_to_tokens(i)
                if token is None:
                    token = f"<tok_{i}>"
                vf.write(token + "\n")
        print(f"  Vocab exported: {vocab_path}")
    except Exception as e:
        print(f"  Vocab export failed: {e}")
    
    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Size: {total_bytes/1e6:.1f} MB ({total_bytes/1e9:.2f} GB)")
    print(f"   Compression: {embed.shape[0]*hidden*4*2/total_bytes:.1f}x vs FP32")
    
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HF model to Lila format")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--output", default="model.lila", help="Output file path")
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    convert(args.model, args.output, args.group_size)
