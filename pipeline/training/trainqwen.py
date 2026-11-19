import logging
from pathlib import Path
from PIL import Image


import torch
from torch.utils.data import DataLoader

# from transformers import get_constant_schedule_with_warmup
from language_model.qwen import Qwen
from pipeline.training.config import ProjectorTrainConfigQwen
from pipeline.training.feature_dataset import FeatureDataset
from projector.two_layer_mlp import MLP

from vision_model.model import SigLIP2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train")

c = ProjectorTrainConfigQwen()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, c.target_rms).to(device)
llm =   Qwen(model_name=c.llm_model_name, device=device)

train_data = FeatureDataset(c.train_dir)
eval_data = FeatureDataset(c.eval_dir)

def loader(data):
    return DataLoader(data, batch_size=c.batch_size, shuffle=True)


train_load = loader(train_data) 
eval_load = loader(eval_data)

prompt_ids = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": c.instruction}],
     add_generation_prompt=True, return_tensors="pt")["input_ids"].to(device)


optimizer = torch.optim.AdamW([
    {"params": mlp.parameters(),       "lr": 1e-4},
    {"params": llm.model.parameters(), "lr": 2e-5},
])

llm.model.train()



for e in range(c.epochs):
    train_loss = 0
    for batch_num, batch in enumerate(train_load):

        features = batch["features"].to(device).float()
        captions = batch["caption"]

        B = features.shape[0]
        P = prompt_ids.shape[1]

        eos = llm.tokenizer.eos_token # <turn|>

        tok = llm.tokenizer(
            [cap + eos for cap in captions],
            add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt").to(device)

        caption_ids = tok.input_ids
        caption_mask = tok.attention_mask
                        
        ids = torch.cat([prompt_ids.expand(B, -1), caption_ids], dim=1) # copy prompt B times
        text = llm.embedding(ids)
        image = mlp(features).to(text.dtype)
        inputs = torch.cat([image, text], dim=1)

        prefix = image.shape[1] + P
        attention_mask = torch.cat([torch.ones(B, prefix, dtype=torch.long, device=device), caption_mask], dim=1)

        labels = caption_ids.masked_fill(caption_mask == 0, -100) # ignore gradients for non-captions
        labels = torch.cat([torch.full((B,prefix), -100, device=device), labels], dim=1)

        # Qwen loss
     
        Qwen_loss = llm.model(inputs_embeds=inputs, attention_mask=attention_mask, labels=labels).loss
        # mlp_loss.backward()
        Qwen_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss  += Qwen_loss.item()
        #print(f"{e+1:>6} | {batch_num+1:>6} | {Qwen_loss.item()/train_load}")
    print(f"epoch {e+1} | training loss {train_loss / len(train_load):.4f}")
    llm.model.eval()
    mlp.eval()
    val_total = 0.0
    with torch.no_grad():
        for batch in eval_load:
            features = batch["features"].to(device).float()
            captions = batch["caption"]
            B = features.shape[0]

            tok = llm.tokenizer([cap + eos for cap in captions],
                add_special_tokens=False, padding=True, padding_side="right", return_tensors="pt").to(device)

            ids = torch.cat([prompt_ids.expand(B, -1), tok.input_ids], dim=1)
            text = llm.embedding(ids)
            inputs = torch.cat([mlp(features).to(text.dtype), text], dim=1)

            attention_mask = torch.cat([torch.ones(B, prefix, dtype=torch.long, device=device), tok.attention_mask], dim=1)
            labels = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
            labels = torch.cat([torch.full((B, prefix), -100, device=device), labels], dim=1)

            val_total += llm.model(inputs_embeds=inputs, attention_mask=attention_mask,
                                   labels=labels).loss.item()

    print(f"epoch {e+1} | validation loss {val_total / len(eval_load):.4f}")

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

 


    

vm = SigLIP2(model_name=c.vision_model_name, device=device)

llm.model.eval()
mlp.eval()

image_path = Path("test_images/shuggibuggidubbi.png")
prompt = ["Describe the man in the image",
            "How many eyes doe this paricular man have",
            "Does this man look happy or sad?",
            "What ethnic race is this man?"]

with torch.no_grad():
    img = Image.open(image_path).convert("RGB")
    features = vm.embed(img)
    img_emb = mlp(features.float())

    prompt_ids = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt[2]}],
        add_generation_prompt=True, return_tensors="pt")["input_ids"].to(device)
    txt_emb = llm.embedding(prompt_ids)


    inputs = torch.cat([img_emb.to(txt_emb.dtype), txt_emb], dim=1)

print("\nprompt: ", prompt)
print("generated:", llm.generate_and_decode(inputs, max_new_tokens=300))




    