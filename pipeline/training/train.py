import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

# from transformers import get_constant_schedule_with_warmup
from language_model.mimir import Mimir
from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset
from projector.two_layer_mlp import MLP
from vision_model.model import SigLIP2
#from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR



logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("train")

c = ProjectorTrainConfig()
LOG_EVERY = 1
running_loss = 0.0
max_batch = 10

mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, c.target_rms).to(c.device)
llm = Mimir(model_name=c.llm_model_name, device=c.device)

# from peft import LoraConfig, get_peft_model
# llm.model = get_peft_model(llm.model, LoraConfig(r=16, lora_alpha=32,
#                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05))

# git pull && uv run pipeline/training/train.py

train_data = FeatureDataset(c.train_dir)
eval_data = FeatureDataset(c.eval_dir)

def loader(data):
   return DataLoader(data, batch_size=c.batch_size, shuffle=True)

train_load = loader(train_data) 
eval_load = loader(eval_data)

# import inspect
# print("token_type_ids" in inspect.signature(llm.model.forward).parameters)

print(f"running {c.num_epochs} epocs")

def build_batch(prompts, captions, features):
    B = features.shape[0]
    seqs, label_seqs = [], []
    for p, cap in zip(prompts, captions):
        p_ids = llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True)["input_ids"]
        c_ids = llm.tokenizer(cap + llm.tokenizer.eos_token, add_special_tokens=False).input_ids
        seqs.append(p_ids + c_ids)
        label_seqs.append([-100] * len(p_ids) + c_ids)

    L = max(map(len,seqs))
    pad = llm.tokenizer.pad_token_id
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=c.device)
    text_mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=c.device)
    text_labels = torch.tensor([s + [-100] * (L - len(s)) for s in label_seqs], device=c.device)

    n_head = 4
    text = llm.embedding(ids)
    img = mlp(features).to(text.dtype)
    n_img = img.shape[1]

    
    inputs = torch.cat([text[:, :n_head], img, text[:, n_head:]], dim=1)
    ones = torch.ones(B, n_img, dtype=torch.long, device=c.device)
    attention_mask = torch.cat([text_mask[:, :n_head], ones, text_mask[:, n_head:]], dim=1)
    no_loss = torch.full((B, n_img), -100, device=c.device)
    labels = torch.cat([text_labels[:, :n_head], no_loss, text_labels[:, n_head:]], dim=1)
    
    # indicator matrix for mimir
    token_type_ids = torch.zeros_like(attention_mask)
    token_type_ids[:, n_head:n_head + n_img] = 1

    return inputs, attention_mask, labels, token_type_ids

def val_loss(shuffle=False, n_batches=50):
    llm.model.eval(); mlp.eval()
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

optimizer = torch.optim.AdamW([
    {"params": mlp.parameters(),       "lr": 1e-4, "weight_decay": 0.01},
    {"params": llm.model.parameters(), "lr": 2e-5, "weight_decay": 0.0},
])

total_steps = len(train_load) * c.num_epochs
llm_warmup = int(0.3 * total_steps)          # 30% of training

# scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=int(0.05 * total_steps),
#     num_training_steps=total_steps
# )

scheduler = LambdaLR(optimizer, lr_lambda=[
    lambda step: 1.0,                                    # full lr from the start
    lambda step: min(1.0, step / max(1, llm_warmup)),    # linear ramp-up
])



llm.model.train()

header = f"{'epoch':>6} │ {'batch':>6} │ {'loss':>8}"
print(header)
print("─" * len(header))

import csv, time
log_path = c.log_dir / f"log_{time.strftime('%Y%m%d-%H%M%S')}.csv"
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr"])

def log_row(*row):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

for e in range(c.num_epochs):
    for batch_num, batch in enumerate(train_load):
       
        features = batch["features"].to(c.device).float()
        inputs, attention_mask, labels, token_type_ids = build_batch(batch["prompt"], batch["caption"], features)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        # Mimir loss
        mimir_loss = llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                               labels=labels, token_type_ids=token_type_ids).loss

        mimir_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(mlp.parameters()) + list(llm.model.parameters()), 1.0)

        
        if (batch_num + 1) % LOG_EVERY == 0:
                    g_mlp = sum(p.grad.norm()**2 for p in mlp.parameters() if p.grad is not None)**0.5
                    g_llm = sum(p.grad.norm()**2 for p in llm.model.parameters() if p.grad is not None)**0.5
                    print(f"grad norm  mlp {g_mlp:.4f}  llm {g_llm:.4f}")

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += mimir_loss.item()

        if (batch_num + 1) % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            print(f"{e+1:>6} | {batch_num+1:>6} | {avg:>8.4f}")
            log_row(e + 1, batch_num + 1, avg, "") #, scheduler.get_last_lr()[0])
            running_loss = 0.0

        out_c = out - out.mean(dim=0, keepdim=True)
        cos_c = torch.nn.functional.cosine_similarity(out_c[0], out_c[1], dim=0)
        print(f"SigLIP cos {cos_in:.3f}  MLP cos {cos_out:.3f}  MLP centered {cos_c:.3f}")

        if batch_num >= max_batch:
            break

    llm.model.eval()
    mlp.eval()
    val_total = 0.0
    with torch.no_grad():
        for batch in eval_load:
            features = batch["features"].to(c.device).float()
            inputs, attention_mask, labels, token_type_ids = build_batch(batch["prompt"], batch["caption"], features)

            val_total += llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                                   labels=labels, token_type_ids=token_type_ids).loss.item()

    avg_val = val_total / len(eval_load)

    v_ok = val_loss()
    v_shuffle = val_loss(shuffle=True)
    print(f"epoch {e+1}/{c.num_epochs} │ val correct {v_ok:.4f} │ val shuffled {v_shuffle:.4f} │ gap {v_shuffle - v_ok:.4f}")
    
    with torch.no_grad():
        b = next(iter(eval_load))
        f = b["features"].to(c.device).float()
        out = mlp(f).flatten(1)
        cos_in  = torch.nn.functional.cosine_similarity(f.flatten(1)[0], f.flatten(1)[1], dim=0)
        cos_out = torch.nn.functional.cosine_similarity(out[0], out[1], dim=0)
        print(f"SigLIP cos {cos_in:.3f}  MLP cos {cos_out:.3f}")

    from pipeline.training.judge import judge_val
    acc, cov = judge_val(llm, mlp, eval_data, c)
    print(f"epoch {e+1}/{c.num_epochs} │ val loss {avg_val:.4f} │ judge accuracy {acc:.2f} │ coverage {cov:.2f}")    
    log_row(e + 1, "", "", avg_val, "", acc, cov)
    print("─" * len(header))
    
    

vm = SigLIP2(model_name=c.vision_model_name, device=c.device)

# image_path = Path("test_images/shuggibuggidubbi.png")
image_path = Path("test_images/shuggibuggidubbi.png")
prompts = [
    "Describe the image.",
    "What is the person in the image doing?",
    "What is the person wearing?",
    "What objects are on the desk?",
    "Is the person smiling?",
    "What can you see through the window?",
    "What color are the curtains?",
    "Is the image taken indoors or outdoors?",
    "What kind of room is this?",
    "What is the lighting like in this image?",
    "What ethnic race is this man?"

]

with torch.no_grad():
    img = Image.open(image_path).convert("RGB")
    img_emb = mlp(vm.embed(img).float())

    for p in prompts:
        prompt_ids = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, return_tensors="pt")["input_ids"].to(c.device)
        txt_emb = llm.embedding(prompt_ids)
        inputs = torch.cat([txt_emb[:, :4], img_emb.to(txt_emb.dtype), txt_emb[:, 4:]], dim=1)

        print(f"\n{'prompt:':<11}{p}")
        print(f"{'generated:':<11}{llm.generate_and_decode(inputs, max_new_tokens=300, n_head=4)}")