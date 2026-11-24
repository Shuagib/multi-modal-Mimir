import logging
from pathlib import Path

import torch
from PIL import Image

# from transformers import get_constant_schedule_with_warmup
# from transformers import get_cosine_schedule_with_warmup
from transformers import get_cosine_with_min_lr_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from language_model.mimir import Mimir
from pipeline.training.build_batch import build_batch
from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset
from pipeline.training.tests.image_gap import image_gap
from projector.two_layer_mlp import MLP
from vision_model.model import SigLIP2
from pipeline.training.judge import judge_val

#from pipeline.training.tests.val_loss import val_loss

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("train")

c = ProjectorTrainConfig()
LOG_EVERY = 100
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

from adam_atan2_pytorch import AdamAtan2

optimizer = AdamAtan2(
    list(mlp.parameters()) + list(llm.model.parameters()),
    lr=1e-5, weight_decay=0.1
)

total_steps = len(train_load) * c.num_epochs
warmup = min(500, int(0.1 * total_steps))     # 500 steps or 10% of total

scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup,
    num_training_steps=total_steps,
    min_lr_rate=0.1,
)

llm.model.train()



import csv
import time

log_path = c.log_dir / f"log_{time.strftime('%Y%m%d-%H%M%S')}.csv"
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "lr"])

def log_row(*row):
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(row)


mlp_params = sum(p.numel() for p in mlp.parameters())
llm_params = sum(p.numel() for p in llm.model.parameters())

run_info = {
    "data":          f"data-{c.TRAIN_COUNT}",
    "train samples": len(train_data),
    "eval samples":  len(eval_data),
    "batch size":    c.batch_size,
    "epochs":        c.num_epochs,
    "steps/epoch":   len(train_load),
    "total steps":   total_steps,
    "warmup steps":  warmup,
    "optimizer":     type(optimizer).__name__,
    "lr":            optimizer.param_groups[0]["lr"] if not scheduler else scheduler.base_lrs[0],
    "weight decay":  optimizer.param_groups[0].get("weight_decay"),
    "MLP bias":      f"layer1={mlp.layer1.bias is not None}, layer2={mlp.layer2.bias is not None}",
    "MLP params":    f"{mlp_params / 1e6:.1f}M",
    "LLM params":    f"{llm_params / 1e6:.0f}M (all trainable)",
    "target_rms":    c.target_rms,
    "LLM":           c.llm_model_name,
    "vm":        c.vision_model_name,
    "device":        str(c.device),
}

def print_table(info: dict, title: str = "run config"):
    kw = max(len(k) for k in info)
    vw = max(len(str(v)) for v in info.values())
    vw = max(vw, len(title) - kw - 3)

    print(f"┌{'─' * (kw + 2)}┬{'─' * (vw + 2)}┐")
    print(f"│ {title:<{kw + vw + 3}} │")
    print(f"├{'─' * (kw + 2)}┼{'─' * (vw + 2)}┤")
    for k, v in info.items():
        print(f"│ {k:<{kw}} │ {str(v):<{vw}} │")
    print(f"└{'─' * (kw + 2)}┴{'─' * (vw + 2)}┘")

print_table(run_info)

header = f"{'epoch':>6} │ {'batch':>6} │ {'loss':>8}"
print(header)
print("─" * len(header))

for e in range(c.num_epochs):
    running_loss = 0.0
    for batch_num, batch in enumerate(train_load):
       
        features = batch["features"].to(c.device).float()
        inputs, attention_mask, labels, token_type_ids = build_batch(batch["prompt"], batch["caption"], features, llm, mlp, c)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        # Mimir loss
        mimir_loss = llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                               labels=labels, token_type_ids=token_type_ids).loss

        mimir_loss.backward()
        # torch.nn.utils.clip_grad_norm_(list(mlp.parameters()) + list(llm.model.parameters()), 1.0)
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(llm.model.parameters(), 1.0)
        
        # if (batch_num + 1) % LOG_EVERY == 0:
        #             g_mlp = sum(p.grad.norm()**2 for p in mlp.parameters() if p.grad is not None)**0.5
        #             g_llm = sum(p.grad.norm()**2 for p in llm.model.parameters() if p.grad is not None)**0.5
        #             print(f"grad norm  mlp {g_mlp:.4f}  llm {g_llm:.4f}")

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += mimir_loss.item()

        if (batch_num + 1) % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            print(f"{e+1:>6} | {batch_num+1:>6} | {avg:>8.4f}")
            log_row(e + 1, batch_num + 1, avg, "") #, scheduler.get_last_lr()[0])
            running_loss = 0.0
        
        # if batch_num >= max_batch:
        #     break

    llm.model.eval()
    mlp.eval()
    val_total = 0.0
    with torch.no_grad():
        for batch in eval_load:
            features = batch["features"].to(c.device).float()
            inputs, attention_mask, labels, token_type_ids = build_batch(batch["prompt"], batch["caption"], features, llm, mlp, c)

            val_total += llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                                   labels=labels, token_type_ids=token_type_ids).loss.item()

    avg_val = val_total / len(eval_load)

    v_ok, v_shuffle = image_gap(llm, mlp, eval_load, c)
    print(f"epoch {e+1}/{c.num_epochs} │ val correct {v_ok:.4f} │ val shuffled {v_shuffle:.4f} │ gap {v_shuffle - v_ok:.4f}")
    

    
    acc, cov = judge_val(llm, mlp, eval_data, c)
    print(f"epoch {e+1}/{c.num_epochs} │ val loss {avg_val:.4f} │ judge accuracy {acc:.2f} │ coverage {cov:.2f}")    
    log_row(e + 1, "", "", avg_val, "", acc, cov)
    print("─" * len(header))
    
c.checkpoint_dir.mkdir(parents=True, exist_ok=True)
torch.save(mlp.state_dict(), c.checkpoint_dir / "mlp.pt")



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
