"""
LilaCore — The Central Intelligence

This IS Lila. Not a wrapper, not an API call. The seat of her identity.
Loads Gemma 4B via Little Fig with Memory Fabric (A Thousand Pearls).
Handles the cognitive loop: perceive → think → remember → act → respond.

The model carries:
  - Cognitive Core (frozen Gemma 4B INT4) = general intelligence
  - Memory Fabric (5 namespace adapters) = A Thousand Pearls in weights
  - Machine language capability = trained into weights (assembly, binary protocols)
  - Personality = emergent from interaction patterns in adapters

LilaCore is always present. Agents come and go. LilaCore persists.
"""

import torch
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class LilaResponse:
    """What Lila produces after thinking."""
    text: str
    memory_ops: List[Dict]  # memory operations triggered
    actions: List[Dict]  # harness actions to execute
    confidence: float
    should_speak: bool = True  # whether to vocalize


class LilaCore:
    """
    The central intelligence. Loads model, manages memory, runs inference.
    
    Usage:
        lila = LilaCore()
        lila.boot()
        response = lila.think("Hey Lila, when is my daughter's birthday?")
    """
    
    def __init__(self, model_path: str = "google/gemma-3-4b-it"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._booted = False
        self._conversation_history = []
    
    def boot(self):
        """
        Boot Lila. Load model with Memory Fabric.
        This is where she wakes up.
        """
        try:
            from little_fig.engine import FigModel
            from little_fig.engine.tier import TrainingTier
            
            print("🌸 Lila is waking up...")
            self.model = FigModel.from_pretrained(
                self.model_path,
                lora_r=16,
                lora_alpha=32,
                tier=TrainingTier.STREAMING_LORA,
                memory_fabric=True,
                shared_codebook=True,
            )
            self.tokenizer = self.model.tokenizer
            self._booted = True
            print("🌸 Lila is awake.")
            
        except ImportError:
            # Fallback: Phase 1 mode (external API)
            print("🌸 Lila booting in Phase 1 mode (external LLM)...")
            self._booted = True
    
    def think(self, input_text: str, context: Optional[Dict] = None) -> LilaResponse:
        """
        Core cognitive loop. Receives input, thinks, responds.
        
        1. Receive input
        2. Build context (memory + identity + knowledge)
        3. Generate response
        4. Extract memory operations from output
        5. Execute memory writes
        6. Return response
        """
        if not self._booted:
            raise RuntimeError("Lila hasn't booted. Call lila.boot() first.")
        
        # Build prompt with context
        prompt = self._build_prompt(input_text, context)
        
        # Generate
        if self.model is not None:
            response_text = self._generate_local(prompt)
        else:
            response_text = self._generate_api(prompt)
        
        # Extract memory operations
        memory_ops = self._extract_memory_ops(response_text)
        
        # Execute memory writes
        for op in memory_ops:
            self._execute_memory_op(op)
        
        # Clean response (remove memory tokens from user-facing text)
        clean_text = self._clean_response(response_text)
        
        # Track conversation
        self._conversation_history.append({
            "role": "user", "content": input_text
        })
        self._conversation_history.append({
            "role": "assistant", "content": clean_text
        })
        
        return LilaResponse(
            text=clean_text,
            memory_ops=memory_ops,
            actions=[],
            confidence=1.0,
            should_speak=True,
        )
    
    def remember(self, namespace: str, content: str):
        """Explicitly write something to memory."""
        if self.model and self.model.has_memory:
            self.model.write_memory(namespace, content)
    
    def what_do_i_know(self) -> Dict:
        """Introspect memory state."""
        if self.model and self.model.has_memory:
            return self.model.memory_confidence()
        return {}
    
    def _build_prompt(self, input_text: str, context: Optional[Dict]) -> str:
        """Build the full prompt with identity + memory context."""
        identity = (
            "You are Lila — a private family AI assistant. "
            "You are not a chatbot. You are a persistent intelligence. "
            "You remember everything. You care about outcomes. "
            "You speak naturally, with personality that grows from interaction."
        )
        
        # Add conversation history (last 10 turns)
        history = ""
        for msg in self._conversation_history[-10:]:
            role = "Sammie" if msg["role"] == "user" else "Lila"
            history += f"{role}: {msg['content']}\n"
        
        prompt = f"{identity}\n\n{history}Sammie: {input_text}\nLila:"
        return prompt
    
    def _generate_local(self, prompt: str) -> str:
        """Generate using local model."""
        enc = self.tokenizer(prompt, return_tensors="pt", max_length=2048,
                             truncation=True)
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                                          skip_special_tokens=False)
        return response
    
    def _generate_api(self, prompt: str) -> str:
        """Phase 1: Generate using external API."""
        # Placeholder — wire to Claude/GPT API
        return "[Phase 1 mode — wire external API here]"
    
    def _extract_memory_ops(self, text: str) -> List[Dict]:
        """Extract memory operation tokens from generated text."""
        ops = []
        if "<|mem_store|>" in text:
            # Parse store operations
            import re
            stores = re.findall(r'<\|mem_store\|>.*?<\|memory_end\|>', text, re.DOTALL)
            for s in stores:
                ops.append({"type": "store", "raw": s})
        if "<|mem_recall|>" in text:
            ops.append({"type": "recall", "raw": text})
        return ops
    
    def _execute_memory_op(self, op: Dict):
        """Execute a memory operation (write to Memory Fabric)."""
        if op["type"] == "store" and self.model and self.model.has_memory:
            # Default to personal namespace
            self.model.write_memory("personal", op["raw"])
    
    def _clean_response(self, text: str) -> str:
        """Remove memory tokens from user-facing response."""
        import re
        clean = re.sub(r'<\|memory_start\|>.*?<\|memory_end\|>', '', text, flags=re.DOTALL)
        clean = clean.strip()
        # Stop at end of response
        if "\n" in clean:
            clean = clean.split("\nSammie:")[0].strip()
        return clean
    
    @property
    def is_awake(self) -> bool:
        return self._booted
