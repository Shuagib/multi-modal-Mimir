import torch
from pipeline.training.build_batch import build_batch

def image_gap(llm, mlp, eval_load, c, n_batches=50):
    llm.model.eval()
    mlp.eval()
    ok_total, shuf_total, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(eval_load):
            if i >= n_batches: break
            f = batch["features"].to(c.device).float()
            for shuffle in (False, True):
                ff = f.roll(1, dims=0) if shuffle else f
                inputs, am, labels, tti = build_batch(batch["prompt"], batch["caption"], ff, llm, mlp, c)
                loss = llm.model(inputs_embeds=inputs, attention_mask=am,
                                 labels=labels, token_type_ids=tti).loss.item()
                if shuffle: shuf_total += loss
                else:       ok_total  += loss
            n += 1
    llm.model.train(); mlp.train()
    return ok_total / n, shuf_total / n