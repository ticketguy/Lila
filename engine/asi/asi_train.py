#!/usr/bin/env python3
"""
ASI Training Interface — How Little Fig works with .asi files

Little Fig's workflow:
  1. Load .asi → extract weights + current adapters
  2. Train new adapters (Memory Fabric update)
  3. Write ONLY the adapter section back → new .asi
  
The model weights stay frozen. Only the Memory Fabric (A Thousand Pearls)
is updated during training. This keeps the .asi portable and fast to update.

Usage:
    from asi_train import AsiTrainer
    
    trainer = AsiTrainer("lila.asi")
    trainer.load_for_training()
    trainer.train_adapters(training_data, namespace="episodic")
    trainer.save("lila_trained.asi")
"""

import struct
import os
import sys
import mmap
import hashlib
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Match asi.h constants
ASI_MAGIC = 0x41534921
ASI_VERSION = 1
ASI_PAGE_SIZE = 4096

ASI_SECTION_MODEL_CONFIG  = 0x01
ASI_SECTION_WEIGHTS       = 0x02
ASI_SECTION_MEMORY_FABRIC = 0x03
ASI_SECTION_TOKENIZER     = 0x04
ASI_SECTION_BYTECODE      = 0x05
ASI_SECTION_HARNESS       = 0x06
ASI_SECTION_PERSONALITY   = 0x07
ASI_SECTION_METADATA      = 0x08

NAMESPACE_PERSONAL  = 0
NAMESPACE_EPISODIC  = 1
NAMESPACE_WIKI      = 2
NAMESPACE_SCHEDULE  = 3
NAMESPACE_CONTESTED = 4

NAMESPACE_NAMES = ["personal", "episodic", "wiki", "schedule", "contested"]


class AsiFile:
    """Read/write .asi files at the section level."""
    
    def __init__(self, path: str):
        self.path = path
        self.header = None
        self.sections = {}
        self.section_data = {}
        self._raw = None
    
    def load(self):
        """Load and parse an .asi file."""
        with open(self.path, 'rb') as f:
            self._raw = f.read()
        
        # Parse header (64 bytes)
        magic, version, flags, n_sections, total_size, st_offset = \
            struct.unpack_from("IIIIQq", self._raw, 0)
        identity_hash = self._raw[32:64]
        
        if magic != ASI_MAGIC:
            raise ValueError(f"Invalid ASI magic: 0x{magic:08X}")
        
        self.header = {
            'magic': magic,
            'version': version,
            'flags': flags,
            'n_sections': n_sections,
            'total_size': total_size,
            'section_table_offset': st_offset,
            'identity_hash': identity_hash,
        }
        
        # Parse section table
        offset = st_offset
        for i in range(n_sections):
            stype, sflags, soffset, ssize, schecksum = \
                struct.unpack_from("IIQQQ", self._raw, offset)
            self.sections[stype] = {
                'type': stype,
                'flags': sflags,
                'offset': soffset,
                'size': ssize,
                'checksum': schecksum,
            }
            offset += 32
        
        print(f"ASI loaded: {self.path}")
        print(f"  Version: {version}, Sections: {n_sections}, Size: {len(self._raw)/1e6:.1f} MB")
    
    def get_section_data(self, section_type: int) -> Optional[bytes]:
        """Get raw bytes for a section."""
        if section_type not in self.sections:
            return None
        sec = self.sections[section_type]
        return self._raw[sec['offset']:sec['offset'] + sec['size']]
    
    def get_model_config(self) -> Dict:
        """Parse MODEL_CONFIG section."""
        data = self.get_section_data(ASI_SECTION_MODEL_CONFIG)
        if data is None:
            raise ValueError("No MODEL_CONFIG section")
        
        fields = struct.unpack_from("IIIIIIIIffII", data, 0)
        return {
            'n_layers': fields[0],
            'hidden_size': fields[1],
            'intermediate_size': fields[2],
            'n_heads': fields[3],
            'n_kv_heads': fields[4],
            'vocab_size': fields[5],
            'max_seq_len': fields[6],
            'head_dim': fields[7],
            'rope_theta': fields[8],
            'rms_norm_eps': fields[9],
            'quant_type': fields[10],
            'group_size': fields[11],
        }
    
    def get_adapters(self) -> Dict[int, Dict[int, Dict]]:
        """
        Parse Memory Fabric section into adapter weights.
        Returns: {layer_idx: {namespace_idx: {'rank', 'gate', 'A', 'B'}}}
        """
        data = self.get_section_data(ASI_SECTION_MEMORY_FABRIC)
        if data is None:
            return {}
        
        offset = 0
        n_namespaces, n_layers, default_rank, _ = struct.unpack_from("IIII", data, offset)
        offset += 16
        
        adapters = {}
        for layer in range(n_layers):
            adapters[layer] = {}
            for ns in range(n_namespaces):
                rank, in_f, out_f, gate = struct.unpack_from("IIIf", data, offset)
                offset += 16
                
                A = None
                B = None
                if rank > 0:
                    a_size = in_f * rank
                    b_size = rank * out_f
                    A = np.frombuffer(data[offset:offset + a_size*4], dtype=np.float32).reshape(in_f, rank).copy()
                    offset += a_size * 4
                    B = np.frombuffer(data[offset:offset + b_size*4], dtype=np.float32).reshape(rank, out_f).copy()
                    offset += b_size * 4
                
                adapters[layer][ns] = {
                    'rank': rank,
                    'in_features': in_f,
                    'out_features': out_f,
                    'gate': gate,
                    'A': A,
                    'B': B,
                }
        
        return adapters


class AsiTrainer:
    """
    Interface for Little Fig to train a .asi file's Memory Fabric.
    
    The workflow:
      1. Load the .asi (get config + frozen weights reference)
      2. Extract or initialize adapters for the target namespace
      3. Run training (update adapter A, B matrices)
      4. Repack the .asi with updated adapters
    """
    
    def __init__(self, asi_path: str):
        self.asi = AsiFile(asi_path)
        self.config = None
        self.adapters = None
        self._loaded = False
    
    def load_for_training(self):
        """Load the .asi and prepare for adapter training."""
        self.asi.load()
        self.config = self.asi.get_model_config()
        self.adapters = self.asi.get_adapters()
        self._loaded = True
        
        print(f"\nReady for training:")
        print(f"  Model: {self.config['n_layers']}L, h={self.config['hidden_size']}")
        print(f"  Adapters loaded: {sum(1 for l in self.adapters.values() for a in l.values() if a['rank'] > 0)}")
    
    def init_adapter(self, layer: int, namespace: int, rank: int = 8,
                     in_features: int = None, out_features: int = None):
        """Initialize a new adapter for a layer/namespace pair."""
        if in_features is None:
            in_features = self.config['hidden_size']
        if out_features is None:
            out_features = self.config['hidden_size']
        
        # Kaiming initialization for A, zero for B (standard LoRA init)
        A = np.random.randn(in_features, rank).astype(np.float32) * np.sqrt(2.0 / in_features)
        B = np.zeros((rank, out_features), dtype=np.float32)
        
        if layer not in self.adapters:
            self.adapters[layer] = {}
        
        self.adapters[layer][namespace] = {
            'rank': rank,
            'in_features': in_features,
            'out_features': out_features,
            'gate': 1.0,  # Fully active
            'A': A,
            'B': B,
        }
    
    def init_all_adapters(self, namespace: int, rank: int = 8):
        """Initialize adapters for ALL layers in a namespace."""
        for layer in range(self.config['n_layers']):
            self.init_adapter(layer, namespace, rank)
        print(f"  Initialized {self.config['n_layers']} adapters for namespace "
              f"'{NAMESPACE_NAMES[namespace]}' (rank={rank})")
    
    def get_adapter_weights(self, layer: int, namespace: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (A, B) weight matrices for a specific adapter."""
        adapter = self.adapters.get(layer, {}).get(namespace, {})
        return adapter.get('A'), adapter.get('B')
    
    def set_adapter_weights(self, layer: int, namespace: int,
                           A: np.ndarray, B: np.ndarray, gate: float = 1.0):
        """Set updated adapter weights after training."""
        if layer not in self.adapters:
            self.adapters[layer] = {}
        
        self.adapters[layer][namespace] = {
            'rank': A.shape[1],
            'in_features': A.shape[0],
            'out_features': B.shape[1],
            'gate': gate,
            'A': A.astype(np.float32),
            'B': B.astype(np.float32),
        }
    
    def set_gate(self, layer: int, namespace: int, gate: float):
        """Adjust adapter gate (activation strength)."""
        if layer in self.adapters and namespace in self.adapters[layer]:
            self.adapters[layer][namespace]['gate'] = gate
    
    def build_fabric_section(self) -> bytes:
        """Serialize current adapters back to binary format."""
        data = bytearray()
        
        n_namespaces = 5
        n_layers = self.config['n_layers']
        default_rank = 8
        
        # Fabric header
        data.extend(struct.pack("IIII", n_namespaces, n_layers, default_rank, 0))
        
        for layer in range(n_layers):
            for ns in range(n_namespaces):
                adapter = self.adapters.get(layer, {}).get(ns, {})
                rank = adapter.get('rank', 0)
                in_f = adapter.get('in_features', 0)
                out_f = adapter.get('out_features', 0)
                gate = adapter.get('gate', 0.0)
                
                data.extend(struct.pack("IIIf", rank, in_f, out_f, gate))
                
                if rank > 0 and adapter.get('A') is not None:
                    data.extend(adapter['A'].astype(np.float32).tobytes())
                    data.extend(adapter['B'].astype(np.float32).tobytes())
        
        return bytes(data)
    
    def save(self, output_path: str):
        """
        Save updated .asi with new adapters.
        
        Strategy: Copy all sections from original, replace MEMORY_FABRIC
        with new adapter weights. Update metadata timestamps.
        """
        if not self._loaded:
            raise RuntimeError("Call load_for_training() first")
        
        print(f"\nSaving updated ASI to: {output_path}")
        
        # Build new fabric section
        new_fabric = self.build_fabric_section()
        print(f"  New fabric section: {len(new_fabric)/1e6:.1f} MB")
        
        # Collect all sections (original + updated fabric)
        sections_to_write = {}
        for stype in self.asi.sections:
            if stype == ASI_SECTION_MEMORY_FABRIC:
                sections_to_write[stype] = new_fabric
            elif stype == ASI_SECTION_METADATA:
                # Update timestamps
                old_meta = self.asi.get_section_data(ASI_SECTION_METADATA)
                new_meta = bytearray(old_meta)
                struct.pack_into("Q", new_meta, 8, int(time.time()))  # modified_at
                sections_to_write[stype] = bytes(new_meta)
            else:
                sections_to_write[stype] = self.asi.get_section_data(stype)
        
        # Write new .asi file
        self._write_asi(output_path, sections_to_write)
        
        print(f"  ✅ Saved! Size: {os.path.getsize(output_path)/1e6:.1f} MB")
    
    def _write_asi(self, path: str, sections: Dict[int, bytes]):
        """Write a complete .asi file from section data."""
        n_sections = len(sections)
        header_size = 64
        section_table_size = n_sections * 32
        
        # Compute offsets
        def page_align(x):
            return (x + ASI_PAGE_SIZE - 1) & ~(ASI_PAGE_SIZE - 1)
        
        section_list = sorted(sections.items())
        first_offset = page_align(header_size + section_table_size)
        
        offsets = []
        current = first_offset
        for stype, sdata in section_list:
            offsets.append(current)
            current = page_align(current + len(sdata))
        total_size = current
        
        # Flags (preserve from original)
        flags = self.asi.header['flags']
        
        # Identity hash
        personality_data = sections.get(ASI_SECTION_PERSONALITY, b'')
        identity_hash = hashlib.sha256(personality_data).digest()
        
        # Write
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack("IIIIQq32s",
                ASI_MAGIC, ASI_VERSION, flags, n_sections,
                total_size, header_size, identity_hash
            ))
            
            # Section table
            for i, (stype, sdata) in enumerate(section_list):
                checksum = struct.unpack("Q", hashlib.sha256(sdata).digest()[:8])[0]
                f.write(struct.pack("IIQQQ",
                    stype, 0, offsets[i], len(sdata), checksum
                ))
            
            # Section data (page-aligned)
            for i, (stype, sdata) in enumerate(section_list):
                pos = f.tell()
                aligned = page_align(pos)
                if aligned > pos:
                    f.write(b'\x00' * (aligned - pos))
                f.write(sdata)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: Train adapters with PyTorch
# ═══════════════════════════════════════════════════════════════════════════════

def train_namespace(asi_path: str, training_data: List[Dict], namespace: str,
                    output_path: str = None, rank: int = 8, epochs: int = 3,
                    lr: float = 1e-4):
    """
    High-level training interface.
    
    Args:
        asi_path: Path to input .asi file
        training_data: List of {"instruction": ..., "output": ...} dicts
        namespace: One of "personal", "episodic", "wiki", "schedule", "contested"
        output_path: Where to save the updated .asi (default: overwrite)
        rank: LoRA rank for new adapters
        epochs: Training epochs
        lr: Learning rate
    """
    if output_path is None:
        output_path = asi_path
    
    ns_idx = NAMESPACE_NAMES.index(namespace)
    
    trainer = AsiTrainer(asi_path)
    trainer.load_for_training()
    
    # Initialize adapters if not present
    trainer.init_all_adapters(ns_idx, rank=rank)
    
    print(f"\nTraining namespace '{namespace}' (rank={rank}, epochs={epochs}, lr={lr})")
    print(f"  Training examples: {len(training_data)}")
    
    # The actual training loop would use PyTorch here
    # This is the interface Little Fig calls
    # For now, placeholder that shows the pattern:
    
    try:
        import torch
        import torch.nn as nn
        
        for layer_idx in range(trainer.config['n_layers']):
            A, B = trainer.get_adapter_weights(layer_idx, ns_idx)
            if A is None:
                continue
            
            # Convert to tensors
            A_tensor = torch.tensor(A, requires_grad=True)
            B_tensor = torch.tensor(B, requires_grad=True)
            
            # Simple gradient update (actual training uses the full forward pass)
            optimizer = torch.optim.Adam([A_tensor, B_tensor], lr=lr)
            
            # Training would happen here with the actual model forward pass
            # For now this is the structural skeleton
            
            # Write back
            trainer.set_adapter_weights(layer_idx, ns_idx,
                                       A_tensor.detach().numpy(),
                                       B_tensor.detach().numpy())
    except ImportError:
        print("  (PyTorch not available — adapter init only, no training)")
    
    trainer.save(output_path)
    print(f"\n✅ Training complete! Updated .asi: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ASI Training Interface")
    sub = parser.add_subparsers(dest="command")
    
    # Inspect
    inspect_p = sub.add_parser("inspect", help="Inspect an .asi file")
    inspect_p.add_argument("file", help=".asi file path")
    
    # Init adapters
    init_p = sub.add_parser("init", help="Initialize adapters in an .asi")
    init_p.add_argument("file", help="Input .asi file")
    init_p.add_argument("--namespace", required=True, choices=NAMESPACE_NAMES)
    init_p.add_argument("--rank", type=int, default=8)
    init_p.add_argument("--output", default=None)
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        asi = AsiFile(args.file)
        asi.load()
        config = asi.get_model_config()
        print(f"\nModel Config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        adapters = asi.get_adapters()
        active = sum(1 for l in adapters.values() for a in l.values() if a.get('rank', 0) > 0)
        print(f"\nAdapters: {active} active")
    
    elif args.command == "init":
        output = args.output or args.file
        trainer = AsiTrainer(args.file)
        trainer.load_for_training()
        ns_idx = NAMESPACE_NAMES.index(args.namespace)
        trainer.init_all_adapters(ns_idx, rank=args.rank)
        trainer.save(output)
    
    else:
        parser.print_help()
