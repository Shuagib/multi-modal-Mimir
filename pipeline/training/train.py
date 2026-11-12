import logging
import torch
from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader
# from transformers import get_constant_schedule_with_warmup



from language_model.model import Mimir
from projector.model import MLP
from vision_model.model import SigLIP2
from pipeline.training.config import ProjectorTrainConfig
from pipeline.training.feature_dataset import FeatureDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train")

c = ProjectorTrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, c.target_rms).to(device)
llm = Mimir(model_name=c.llm_model_name, device=device)

train_data = FeatureDataset(c.train_dir)
# train_data = Subset(FeatureDataset(c.train_dir), [0])   # only sample 0

loader = DataLoader(train_data, batch_size=c.batch_size, shuffle=True)

prompt_ids = llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": c.instruction}],
     add_generation_prompt=True, return_tensors="pt")["input_ids"].to(device)


optimizer = torch.optim.AdamW([
    {"params": mlp.parameters(),       "lr": 1e-4},
    {"params": llm.model.parameters(), "lr": 2e-5},
])

llm.model.train()

print(f"{'epoch':>6} | {'batch':>6} | {'loss':>8}")
print("-" * 26)

for e in range(c.epochs):
    for batch_num, batch in enumerate(loader):

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

        prefix = 256 + P
        attention_mask = torch.cat([torch.ones(B, prefix, dtype=torch.long, device=device), caption_mask], dim=1)

        labels = caption_ids.masked_fill(caption_mask == 0, -100) # ignore gradients for non-captions
        labels = torch.cat([torch.full((B,prefix), -100, device=device), labels], dim=1)

        # Mimir loss
        token_type_ids = torch.zeros_like(attention_mask)
        token_type_ids[:, :prefix] = 1

        # mlp_loss = mlp.model(inputs_embeds=inputs, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids).loss
        mimir_loss = llm.model(inputs_embeds=inputs, attention_mask=attention_mask, labels=labels, token_type_ids=token_type_ids).loss
        # mlp_loss.backward()
        mimir_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"{e+1:>6} | {batch_num+1:>6} | {mimir_loss.item():>8.4f}")

    llm.model.eval()
    mlp.eval()

    # eval_data = FeatureDataset(c.train_dir)

    vm = SigLIP2(model_name=c.vision_model_name, device=device)

    image_path = Path("test_images/shuggibuggidubbi.png")
    prompt = ["Describe the man in the image",
              "How many eyes doe this paricular man have",
              "Does this man look kappy or sad?",
              "What ethnic race is this man?"]

    with torch.no_grad():
        img = Image.open(image_path).convert("RGB")
        features = vm.embed(img)
        img_emb = mlp(features.float())

        prompt_ids = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt[0]}],
            add_generation_prompt=True, return_tensors="pt")["input_ids"].to(device)

        txt_emb = torch.cat([img.emb.to(txt_emb.dtype), txt_emb], dim=1)

    print("\nprompt: ", prompt)
    print("generated:", llm.generate_and_decode(inputs, max_new_tokens=100))


    # for batch in eval_data:
        # features = batch["features"].unsqueeze(0).to(device).float()   # (1, 256, 768)
