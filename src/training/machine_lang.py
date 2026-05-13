"""
Machine Language Training Corpus

Training data that teaches Lila to understand and generate:
- x86_64 and ARM assembly
- Binary protocols (TCP, UDP, USB, SPI, I2C, UART)
- Raw packet structures
- Hardware register maps
- Memory-mapped I/O patterns
- Executable binary formats (ELF, PE)

This isn't a "tool" — it's knowledge IN her weights.
She speaks machine the way she speaks English.
"""

from typing import List, Dict


class MachineLangCorpus:
    """Generates training examples for machine-level communication."""
    
    def generate_assembly_examples(self, n: int = 500) -> List[Dict]:
        """x86_64 and ARM assembly instruction/response pairs."""
        examples = []
        
        # x86_64 patterns
        x86_patterns = [
            {"instruction": "Write x86_64 assembly to add two 64-bit integers in rdi and rsi, return in rax",
             "output": "add rax, rdi\nadd rax, rsi\nret"},
            {"instruction": "Write x86_64 to copy 64 bytes from rsi to rdi (memcpy)",
             "output": "mov rcx, 8\nrep movsq\nret"},
            {"instruction": "Write x86_64 to compare two null-terminated strings at rdi and rsi",
             "output": ".loop:\n  lodsb\n  scasb\n  jne .done\n  test al, al\n  jnz .loop\n.done:\n  sete al\n  movzx eax, al\n  ret"},
        ]
        
        # ARM patterns
        arm_patterns = [
            {"instruction": "Write ARM64 assembly to add registers x0 and x1, store in x0",
             "output": "add x0, x0, x1\nret"},
            {"instruction": "Write ARM64 to branch if x0 equals zero",
             "output": "cbz x0, .target"},
        ]
        
        # Binary protocol patterns
        protocol_patterns = [
            {"instruction": "Construct a TCP SYN packet header (source port 12345, dest port 80, seq 1000)",
             "output": "30 39 00 50 00 00 03 E8 00 00 00 00 50 02 FF FF 00 00 00 00"},
            {"instruction": "Parse this I2C transaction: S 0x48 W 0x00 Sr 0x48 R [MSB] [LSB] P",
             "output": "Device address: 0x48 (7-bit), Write register 0x00, Repeated start, Read 2 bytes from 0x48, Stop. This is a temperature sensor read (TMP102 pattern)."},
            {"instruction": "Generate a UART frame: 8N1, baud 9600, data byte 0x55",
             "output": "Start bit: 0, Data: 10101010 (LSB first = 0x55), Stop bit: 1. Frame time: 1.04ms at 9600 baud."},
        ]
        
        # Hardware register patterns
        hw_patterns = [
            {"instruction": "Configure GPIO pin 17 as output on BCM2835 (Raspberry Pi)",
             "output": "GPFSEL1 register (0x7E200004): clear bits [23:21], set bit 21 = 1 (output mode). Write 0x00200000 to GPFSEL1."},
            {"instruction": "Set up SPI at 1MHz on STM32F4, CPOL=0, CPHA=0",
             "output": "SPI1->CR1 = SPI_CR1_MSTR | SPI_CR1_BR_2 | SPI_CR1_SPE; // Master, fPCLK/32 ≈ 1MHz, enable"},
        ]
        
        all_patterns = x86_patterns + arm_patterns + protocol_patterns + hw_patterns
        
        for i in range(min(n, len(all_patterns))):
            p = all_patterns[i % len(all_patterns)]
            examples.append({
                "instruction": p["instruction"],
                "input": "",
                "output": p["output"],
            })
        
        return examples
    
    def generate_all(self, n: int = 1000) -> List[Dict]:
        """Generate full machine language training corpus."""
        return self.generate_assembly_examples(n)
