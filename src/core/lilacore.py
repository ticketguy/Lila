"""
LilaCore — The Central Intelligence

Little Fig TRAINS the model (offline, produces weight files).
LilaCore RUNS the trained model (loads weights, does inference).
No Little Fig dependency at runtime.

Loading priority:
  1. Lila Engine (custom assembly — when built)
  2. llama-cpp-python (GGUF format) — fastest existing option
  3. transformers (safetensors) — fallback
  4. External API (Phase 1 only)

Continuous learning cycle:
  Interactions logged → Little Fig trains offline → new weights → Lila hot-reloads
"""

import os
import json
import re
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LilaResponse:
    """What Lila produces after thinking."""
    text: str
    memory_ops: List[Dict]
    actions: List[Dict]
    confidence: float
    should_speak: bool = True


class LilaCore:
    """
    The central intelligence. Loads trained model, runs inference.
    
    Usage:
        lila = LilaCore()
        lila.boot()
        response = lila.think("Hey Lila, what's on my schedule today?")
    """
    
    def __init__(self, model_path: str = None, gguf_path: str = None, api_mode: bool = False):
        self.model_path = model_path
        self.gguf_path = gguf_path
        self.api_mode = api_mode
        self._model = None
        self._tokenizer = None
        self._llm = None
        self._booted = False
        self._conversation_history = []
        self._training_log = []
    
    def boot(self):
        """Wake Lila up. Load the model."""
        print("\U0001f338 Lila is waking up...")
        
        if self.gguf_path and os.path.exists(self.gguf_path):
            self._boot_gguf()
        elif self.model_path and os.path.exists(self.model_path):
            self._boot_transformers()
        elif self.api_mode:
            self._boot_api()
        else:
            default_gguf = os.path.expanduser("~/.lila/model.gguf")
            default_hf = os.path.expanduser("~/.lila/model/")
            if os.path.exists(default_gguf):
                self.gguf_path = default_gguf
                self._boot_gguf()
            elif os.path.exists(default_hf):
                self.model_path = default_hf
                self._boot_transformers()
            else:
                print("\U0001f338 No local model found. Phase 1 (API) mode.")
                self._boot_api()
        
        self._booted = True
        print("\U0001f338 Lila is awake.")
    
    def _boot_gguf(self):
        try:
            from llama_cpp import Llama
            print(f"   Loading GGUF: {self.gguf_path}")
            self._llm = Llama(
                model_path=self.gguf_path, n_ctx=4096,
                n_threads=os.cpu_count(), n_gpu_layers=-1, verbose=False)
            print("   Done (llama.cpp)")
        except ImportError:
            print("   llama-cpp-python not installed. Falling back...")
            self._boot_transformers()
    
    def _boot_transformers(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            path = self.model_path or "google/gemma-3-4b-it"
            print(f"   Loading: {path}")
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            print("   Done (transformers)")
        except Exception as e:
            print(f"   Failed: {e}")
            self._boot_api()
    
    def _boot_api(self):
        self.api_mode = True
        print("   API mode (Phase 1)")
    
    def think(self, input_text: str, context: Optional[Dict] = None) -> LilaResponse:
        """Core cognitive loop."""
        if not self._booted:
            raise RuntimeError("Call lila.boot() first.")
        
        prompt = self._build_prompt(input_text, context)
        
        if self._llm:
            response_text = self._generate_gguf(prompt)
        elif self._model:
            response_text = self._generate_transformers(prompt)
        else:
            response_text = self._generate_api(prompt)
        
        memory_ops = self._extract_memory_ops(response_text)
        clean_text = self._clean_response(response_text)
        self._log_interaction(input_text, clean_text)
        
        self._conversation_history.append({"role": "user", "content": input_text})
        self._conversation_history.append({"role": "assistant", "content": clean_text})
        
        return LilaResponse(text=clean_text, memory_ops=memory_ops,
                           actions=[], confidence=1.0, should_speak=True)
    
    def _build_prompt(self, input_text: str, context: Optional[Dict]) -> str:
        identity = ("You are Lila, Sammie's private family AI assistant. "
                   "Persistent, caring, grows smarter over time. "
                   "Remembers everything. Speaks with warmth and personality.")
        history = ""
        for msg in self._conversation_history[-10:]:
            role = "Sammie" if msg["role"] == "user" else "Lila"
            history += f"{role}: {msg['content']}\n"
        if context and context.get("mode") == "reflection":
            return f"[Internal reflection]\n{input_text}"
        return f"{identity}\n\n{history}Sammie: {input_text}\nLila:"
    
    def _generate_gguf(self, prompt: str) -> str:
        output = self._llm(prompt, max_tokens=512, temperature=0.7, 
                          top_p=0.9, stop=["Sammie:", "\n\n\n"])
        return output["choices"][0]["text"]
    
    def _generate_transformers(self, prompt: str) -> str:
        import torch
        enc = self._tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        device = next(self._model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=512, do_sample=True,
                                       temperature=0.7, top_p=0.9,
                                       pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
    
    def _generate_api(self, prompt: str) -> str:
        return "[Phase 1 — wire API here]"
    
    def _extract_memory_ops(self, text: str) -> List[Dict]:
        ops = []
        if "<|mem_store|>" in text: ops.append({"type": "store", "raw": text})
        if "<|mem_recall|>" in text: ops.append({"type": "recall", "raw": text})
        if "<|mem_conflict|>" in text: ops.append({"type": "conflict", "raw": text})
        return ops
    
    def _clean_response(self, text: str) -> str:
        clean = re.sub(r'<\|memory_start\|>.*?<\|memory_end\|>', '', text, flags=re.DOTALL)
        clean = re.sub(r'<\|mem_\\w+\|>', '', clean)
        clean = clean.split("Sammie:")[0].strip()
        return clean
    
    def _log_interaction(self, user_input: str, lila_response: str):
        entry = {"timestamp": datetime.now().isoformat(), "user": user_input, "assistant": lila_response}
        self._training_log.append(entry)
        log_dir = os.path.expanduser("~/.lila/training_data/")
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "interactions.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def remember(self, namespace: str, content: str):
        self._log_interaction(f"[MEMORY:{namespace}]", content)
    
    def what_do_i_know(self) -> Dict:
        return {"turns": len(self._conversation_history),
                "pending_training": len(self._training_log),
                "engine": "gguf" if self._llm else ("hf" if self._model else "api")}
    
    @property
    def is_awake(self) -> bool:
        return self._booted
