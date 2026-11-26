"""Generate test captions with a saved checkpoint (no training).

usage:  uv run pipeline/training/eval_saved.py <model_saves/run_dir> [n_samples]
        JUDGE=1 uv run ...   also scores with the Qwen judge (slower)
"""
import json
import os
import sys
from pathlib import Path

import torch

from language_model.mimir import Mimir
from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset
from projector.two_layer_mlp import MLP

run = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
c = ProjectorTrainConfig()

# models
llm = Mimir(model_name=str(run / "llm"), device=c.device)
llm.model.eval()
ck = torch.load(run / "mlp.pt", map_location=c.device)
mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, ck["target_rms"]).to(c.device)
mlp.load_state_dict(ck["state_dict"])
mlp.eval()

judge = None
if os.environ.get("JUDGE") == "1":
    from pipeline.training.judge import judge_score as judge

test_data = FeatureDataset(c.test_dir)
out = run / "eval_captions.jsonl"
scores = []
with torch.no_grad(), open(out, "w") as fh:
    for i in range(min(n, len(test_data))):
        s = test_data[i]
        feats = torch.as_tensor(s["features"]).unsqueeze(0).to(c.device).float()
        p_ids = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": s["prompt"]}],
            add_generation_prompt=True, return_tensors="pt")["input_ids"].to(c.device)
        txt = llm.embedding(p_ids)
        img = mlp(feats).to(txt.dtype)
        inputs = torch.cat([txt[:, :4], img, txt[:, 4:]], dim=1)
        gen = llm.generate_and_decode(inputs, max_new_tokens=150)

        r = {"i": i, "prompt": s["prompt"], "reference": s["caption"], "generated": gen}
        if judge:
            r["accuracy"], r["coverage"] = judge(s["caption"], gen, c)
            scores.append(r["accuracy"])
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n── {i} ── {s['prompt']}")
        print(f"REF: {s['caption'][:300]}")
        print(f"GEN: {gen[:300]}", flush=True)

if scores:
    ok = [x for x in scores if x is not None]
    print(f"\njudge acc: {sum(ok) / len(ok):.2f} over {len(ok)}")
print(f"\nsaved {out}")
