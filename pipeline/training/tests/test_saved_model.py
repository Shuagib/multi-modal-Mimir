"""Evaluate a saved run on the test set.

Usage:
    uv run -m pipeline.training.tests.test_saved pipeline/training/model_saves/<run_name>
    uv run -m pipeline.training.tests.test_saved <run_dir> --n 50 --no-judge
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from language_model.mimir import Mimir
from pipeline.training.build_batch import build_batch
from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset
from pipeline.training.tests.image_gap import image_gap
from projector.two_layer_mlp import MLP

def load_mlp(p: Path, c: ProjectorTrainConfig) -> MLP:
    checkpoint = torch.load(p, map_location=c.device)
    state = checkpoint.get("state_dict", checkpoint)
    mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, checkpoint.get("target_rms", c.target_rms))
    mlp.layer1 = nn.Linear(c.mlp_in_dim, c.mlp_hidden_dim, bias="layer1.bias" in state)
    mlp.layer2 = nn.Linear(c.mlp_hidden_dim, c.mlp_out_dim, bias="layer2.bias" in state)
    mlp.load_state_dict(state)

    if isinstance(checkpoint, dict) and "run_info" in checkpoint:
        print("run info:", checkpoint["run_info"])

    return mlp.to(c.device.eval())

@torch.no_grad()
def test_loss(llm, mlp, loader, c) -> float:
    total = 0.0
    for batch in loader:
        features = batch["features"].to(c.device).float()
        inputs, attention_mask, labels, token_type_ids = build_batch(batch["features"], batch["captios"], features, llm, mlp, c)
        total += llm.model(
            input_embeds=inputs,
            attention_mask=attention_mask,
            labels=labels,
            token_type_ids=token_type_ids
            ).loss.item()
    return total / len(loader)

@torch.no_grad()
def generate(llm, mlp, sample, c) -> str:
    features = torch.as_tensor(sample["features"]).unsqueeze(0).to(c.device).float()
    p_ids = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["prompt"]}],
        add_generation_prompt=True, return_tensors="pt")["input_ids"].to(c.device)
    txt = llm.embedding(p_ids)
    img = mlp(features).to(txt.dtype)
    inputs = torch.cat([txt[:, :4], img, txt[:, 4:]], dim=1)
    return llm.generate_and_decode(inputs, max_new_tokens=150)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="folder with mlp.pt (and llm/ if fine-tuned)")
    ap.add_argument("--n", type=int, default=None, help="number of samples to generate (default: all)")
    ap.add_argument("--no-judge", action="store_true", help="skip judge scoring")
    args = ap.parse_args()

    c = ProjectorTrainConfig()
    llm_path = args.run_dir / "llm"
    llm_name = str(llm_path) if llm_path.exists() else c.llm_model_name
    print(f"LLM: {llm_name}")

    llm = Mimir(model_name=llm_name, device=c.device)
    llm.model.eval()
    mlp = load_mlp(args.run_dir / "mlp.pt", c)

    test_data = FeatureDataset(c.test_dir)
    test_load = DataLoader(test_data, batch_size=c.batch_size, shuffle=False)
    print(f"test set: {c.test_dir} ({len(test_data)} samples)")

    loss = test_loss(llm, mlp, test_load, c)
    t_ok, t_shuf = image_gap(llm, mlp, test_load, c, n_batches=len(test_load))
    print(f"test loss {loss:.4f}   image gap {t_shuf - t_ok:+.4f}")

    if not args.no_judge:
        from pipeline.training.judge import judge_score

    n = min(args.n or len(test_data), len(test_data))
    out_path = args.run_dir / "test.jsonl"
    accs, covs = [], []
    with open(out_path, "w") as fh:
        for i in range(n):
            s = test_data[i]
            gen = generate(llm, mlp, s, c)
            acc = cov = None
            if not args.no_judge:
                acc, cov = judge_score(s["caption"], gen, c)
                if acc is not None: accs.append(acc)
                if cov is not None: covs.append(cov)
            fh.write(json.dumps({"i": i, "prompt": s["prompt"], "reference": s["caption"],
                                 "generated": gen, "accuracy": acc, "coverage": cov}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"generated {i + 1}/{n}", flush=True)
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"\n{'samples':12} {n}\n{'test loss':12} {loss:.4f}\n{'image gap':12} {t_shuf - t_ok:+.4f}")
    if not args.no_judge:
        print(f"{'judge acc':12} {avg(accs):.2f}\n{'judge cov':12} {avg(covs):.2f}")
    print(f"captions -> {out_path}")

if __name__ == "__main__":
    main()
