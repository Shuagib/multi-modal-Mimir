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
from transformers import get_cosine_schedule_with_warmup


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train")

c = ProjectorTrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, c.target_rms).to(device)
llm = Mimir(model_name=c.llm_model_name, device=device)

# from peft import LoraConfig, get_peft_model
# llm.model = get_peft_model(llm.model, LoraConfig(r=16, lora_alpha=32,
#                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05))

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
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=device)
    text_mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=device)
    text_labels = torch.tensor([s + [-100] * (L - len(s)) for s in label_seqs], device=device)

    text = llm.embedding(ids)
    inputs = torch.cat([mlp(features).to(text.dtype), text], dim=1)
    n_img = inputs.shape[1] - L
    attention_mask = torch.cat([torch.ones(B, n_img, dtype=torch.long, device=device), text_mask], dim=1)
    labels = torch.cat([torch.full((B,n_img), -100, device=device), text_labels], dim=1)
    
    # mask for 
    token_type_ids = torch.zeros_like(attention_mask)
    token_type_ids[:, :n_img] = 1
    
    return inputs, attention_mask, labels, token_type_ids


optimizer = torch.optim.AdamW([
    {"params": mlp.parameters(),       "lr": 1e-4},
    {"params": llm.model.parameters(), "lr": 2e-5},
], weight_decay=0.05)

total_steps = len(train_load) * c.num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.05 * total_steps),
    num_training_steps=total_steps
)



llm.model.train()

header = f"{'epoch':>6} │ {'batch':>6} │ {'loss':>8}"
print(header)
print("─" * len(header))

old_val = False
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

        features = batch["features"].to(device).float()
        captions = batch["caption"]

        eos = llm.tokenizer.eos_token # <turn|>
        tok = llm.tokenizer(
            [cap + eos for cap in captions],
            add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt").to(device)

        caption_ids = tok.input_ids
        caption_mask = tok.attention_mask        
        # Mimir loss
        features = batch["features"].to(device).float()
        inputs, attention_mask, labels, token_type_ids = build_batch(batch["prompt"], batch["caption"], features)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        mimir_loss = llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                               labels=labels, token_type_ids=token_type_ids).loss

        # mlp_loss.backward()
        mimir_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        print(f"{e+1:>6} | {batch_num+1:>6} | {mimir_loss.item():>8.4f}")
        log_row(e + 1, batch_num + 1, mimir_loss.item(), "", scheduler.get_last_lr()[0])

    llm.model.eval()
    mlp.eval()
    val_total = 0.0
    with torch.no_grad():
        for batch in eval_load:
            features = batch["features"].to(device).float()
            inputs, attention_mask, labels, token_type_ids = build_batch(batch["prompt"], batch["caption"], features)

            val_total += llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                                   labels=labels, token_type_ids=token_type_ids).loss.item()

    new_val = val_total / len(eval_load)

    print("─" * len(header))
    if old_val:
        print(f"epoch {e+1}/{c.num_epochs} │ val loss {val_total / len(eval_load):.4f} │ val diff {old_val - new_val}")
    else: 
        print(f"epoch {e+1}/{c.num_epochs} │ val loss {val_total / len(eval_load):.4f}")
    log_row(e + 1, "", "", new_val, "")
    old_val = new_val
    print("─" * len(header))
    


    llm.model.train()
    mlp.train()
    
    # llm.model.eval()
    # mlp.eval()
    # for i, batch in enumerate(eval_load):
    #     if i >= 3:  # only show a few samples per epoch
    #         break
    #     with torch.no_grad():
    #         features = batch["features"][:1].to(device).float()  # first sample only
    #         img = mlp(features)
    #         txt = llm.embedding(prompt_ids)
    #         inputs = torch.cat([img.to(txt.dtype), txt], dim=1)
    #         generated = llm.generate_and_decode(inputs, max_new_tokens=100)

    #     #print(f"\n--- eval sample {i} ---")
    #     #print("real:     ", batch["caption"][0])
    #     #print("generated:", generated)

    # llm.model.train()
    # mlp.train()

    


    

vm = SigLIP2(model_name=c.vision_model_name, device=device)

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
            add_generation_prompt=True, return_tensors="pt")["input_ids"].to(device)
        txt_emb = llm.embedding(prompt_ids)
        inputs = torch.cat([img_emb.to(txt_emb.dtype), txt_emb], dim=1)

        print(f"\n{'prompt:':<11}{p}")
        print(f"{'generated:':<11}{llm.generate_and_decode(inputs, max_new_tokens=300)}")