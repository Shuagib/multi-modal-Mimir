import torch
from pathlib import Path
from PIL import Image

from vision_model.model import SigLIP2
from projector.model import MLP
from language_model.model import Mimir
from pipeline.utils.image_injection import (insert_special_tokens, inject_patches)

IMAGE_DIR = Path(__file__).resolve().parent.parent / "test_images"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
vm_model_name = "google/siglip2-base-patch16-256"
llm_model_name = "danish-foundation-models/DFM-Mimir-v1.5"

MLP_IN_DIM = 768
MLP_HID_DIM = 2048
MLP_OUT_DIM = 1536

MLP_norm_scale = 0.00906

vm = SigLIP2(model_name=vm_model_name, device=device)
mlp = MLP(MLP_IN_DIM, MLP_HID_DIM, MLP_OUT_DIM, MLP_norm_scale).to(device)
llm = Mimir(model_name=llm_model_name, device=device)

prompts = [
    {"role": "user", "content": "Describe the man in the image"}
]

image_files = [
    "shuggibuggidubbi.png"
]

images = [Image.open(IMAGE_DIR / name) for name in image_files]

for prompt, image  in zip(prompts, images):
    embedded_prompt = llm.embed(prompt) # [1, 20, 1536]
    embedded_image_patches = vm.embed(image) # [1, 256, 768]
    projected_embedded_patches = mlp(embedded_image_patches)
    combined_embedding = inject_patches(projected_embedded_patches, embedded_prompt)
    prompt_embedding = embedded_prompt

    output = llm.generate_and_decode(combined_embedding)
    print(output)

    output = llm.generate_and_decode(prompt_embedding)
    print(output)
