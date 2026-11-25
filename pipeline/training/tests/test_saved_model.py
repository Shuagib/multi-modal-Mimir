"""Run a saved model on a folder of images with randomly chosen prompts.

Usage:
    uv run -m pipeline.training.tests.test_saved_model <run_dir>
    uv run -m pipeline.training.tests.test_saved_model <run_dir> [image_dir] --k 3 --seed 0
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from language_model.mimir import Mimir
from pipeline.training.config import ProjectorTrainConfig
from projector.two_layer_mlp import MLP
from vision_model.model import SigLIP2

DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "test_images"
IMAGE_EXTS ={".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PROMPTS = [
    "Describe the image.",
    "What do you see?",
    "What is in the image?",
    "What colors do you see?",
    "Beskriv billedet.",
]


def load_mlp(p: Path, c: ProjectorTrainConfig) -> MLP:
    checkpoint = torch.load(p, map_location=c.device)
    state = checkpoint.get("state_dict", checkpoint)
    mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, checkpoint.get("target_rms", c.target_rms))
    mlp.layer1 = nn.Linear(c.mlp_in_dim, c.mlp_hidden_dim, bias="layer1.bias" in state)
    mlp.layer2 = nn.Linear(c.mlp_hidden_dim, c.mlp_out_dim, bias="layer2.bias" in state)
    mlp.load_state_dict(state)
    if isinstance(checkpoint, dict) and "run_info" in checkpoint:
        print("run info:", checkpoint["run_info"])
    return mlp.to(c.device).eval()


@torch.no_grad()
def generate(llm: Mimir, mlp: MLP, vm: SigLIP2, image: Image.Image, prompt: str, c: ProjectorTrainConfig) -> str:
    features = vm.embed(image).to(c.device).float()  # [1, 256, 768]
    p_ids = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")["input_ids"].to(c.device)
    txt = llm.embedding(p_ids)
    img = mlp(features).to(txt.dtype)
    inputs = torch.cat([txt[:, :4], img, txt[:, 4:]], dim=1)  # same layout as training
    return llm.generate_and_decode(inputs, max_new_tokens=150)


def find_images(image_dir: Path) -> list[Path]:
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"no images found in {image_dir}")
    return images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="folder with mlp.pt (and llm/ if fine-tuned)")
    ap.add_argument("image_dir", type=Path, nargs="?", default=DEFAULT_IMAGE_DIR,
                    help="folder of images to test on (default: tests/test_images)")
    ap.add_argument("--k", type=int, default=2, help="random prompts per image")
    ap.add_argument("--seed", type=int, default=None, help="fix prompt selection for reproducibility")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    c = ProjectorTrainConfig()
    llm_path = args.run_dir / "llm"
    llm_name = str(llm_path) if llm_path.exists() else c.llm_model_name
    print(f"LLM: {llm_name}")

    llm = Mimir(model_name=llm_name, device=c.device)
    llm.model.eval()
    mlp = load_mlp(args.run_dir / "mlp.pt", c)
    vm = SigLIP2(model_name=c.vision_model_name, device=c.device)

    out_path = args.run_dir / "image_test.jsonl"
    with open(out_path, "w") as fh:
        for path in find_images(args.image_dir):
            image = Image.open(path).convert("RGB")
            show_image(image)
            for prompt in rng.sample(PROMPTS, k=min(args.k, len(PROMPTS))):
                gen = generate(llm, mlp, vm, image, prompt, c)
                print(f"\n[{path.name}] {prompt}\n-> {gen}", flush=True)
                fh.write(json.dumps({"image": path.name, "prompt": prompt, "generated": gen}) + "\n")
    print(f"\nresults -> {out_path}")

def show_image(image: Image.Image, width: int = 400) -> None:
    """Draw an image inline using kitty's graphics protocol."""
    img = image.copy()
    img.thumbnail((width, width))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue())
    chunks = [data[i:i + 4096] for i in range(0, len(data), 4096)]
    for i, chunk in enumerate(chunks):
        more = 1 if i < len(chunks) - 1 else 0
        header = f"a=T,f=100,m={more}" if i == 0 else f"m={more}"
        sys.stdout.write(f"\033_G{header};{chunk.decode()}\033\\")
    sys.stdout.write("\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()

