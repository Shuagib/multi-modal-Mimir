import torch
from pipeline.training.build_batch import build_batch

def val_loss(llm, mlp, eval_load, c, shuffle=False, n_batches=50):
    llm.model.eval()
    mlp.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(eval_load):
            if i >= n_batches: break
            f = batch["features"].to(c.device).float()
            if shuffle:
                f = f.roll(1, dims=0)
            inputs, am, labels, tti = build_batch(batch["prompt"], batch["caption"], f)
            total += llm.model(inputs_embeds=inputs, attention_mask=am,
                               labels=labels, token_type_ids=tti).loss.item()
            n += 1
    llm.model.train(); mlp.train()
    return total / n