import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "mimir": "danish-foundation-models/DFM-Mimir-v1.5",
    "hrm":   "sapientinc/HRM-Text-1B",
    "qwen":  "Qwen/Qwen2.5-1.5B-Instruct",
}

SAMPLE = ("A smiling man in a blue shirt sits at a wooden desk with a laptop "
          "and a coffee mug. Behind him is a window with white curtains.")

caps = [json.loads(l)["caption"] for l in open("data/train/caption-00000.jsonl")][:300]

for key in sys.argv[1:] or MODELS:
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    W = model.get_input_embeddings().weight.detach()

    ids = [i for cap in caps for i in tok(cap, add_special_tokens=False).input_ids]
    print(f"{key:6} whole-matrix RMS: {W.pow(2).mean().sqrt().item():.6f}   "
          f"real-token RMS: {W[ids].pow(2).mean().sqrt().item():.6f}")
    del model, W