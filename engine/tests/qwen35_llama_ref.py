#!/usr/bin/env python3
import numpy as np
from llama_cpp import Llama

MODEL = "/home/ticket/workspaces/lila/models/Qwen3.5-0.8B-heretic-Q4_K_M-imatrix.gguf"
m = Llama(MODEL, n_ctx=64, logits_all=True, verbose=False)
m.eval([9906])
logits = np.array(m.scores[0], dtype=np.float32)
top5 = np.argsort(logits)[-5:][::-1]
print("pos=0 token=9906 top5:", [(int(i), round(float(logits[i]), 5)) for i in top5])

m.eval([11])
logits2 = np.array(m.scores[1], dtype=np.float32)
top5_2 = np.argsort(logits2)[-5:][::-1]
print("pos=1 token=11 top5:", [(int(i), round(float(logits2[i]), 5)) for i in top5_2])

print("ASI generated text:", repr(m.detokenize([9906, 11, 198, 262, 220, 17, 25, 359, 16, 504, 198, 262]).decode("utf-8", "replace")))
m.reset()
out = m.create_completion([9906, 11], max_tokens=10, temperature=0.0)
print("llama_cpp greedy:", repr(out["choices"][0]["text"]))
