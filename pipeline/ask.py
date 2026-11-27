"""Ask the trained model about an image.

one question:   uv run -m pipeline.ask pipeline/training/model_saves/76k_20261127-001436 <image_path> <prompt>

web interface:  uv run --with gradio -m pipeline.ask pipeline/training/model_saves/76k_20261127-001436        (opens http://127.0.0.1:7860)
"""
import sys
from pathlib import Path

import torch
from PIL import Image

from language_model.mimir import Mimir
from pipeline.training.config import ProjectorTrainConfig
from projector.two_layer_mlp import MLP
from vision_model.model import SigLIP2

c = ProjectorTrainConfig()
run = Path(sys.argv[1])

# load on CPU, cast to bf16, then move: fits a 12 GB GPU
llm = Mimir(model_name=str(run / "llm"), device="cpu")
llm.model.to(torch.bfloat16).to(c.device).eval()

ck = torch.load(run / "mlp.pt", map_location=c.device, weights_only=False)
mlp = MLP(c.mlp_in_dim, c.mlp_hidden_dim, c.mlp_out_dim, ck["target_rms"])
mlp.layer1 = torch.nn.Linear(c.mlp_in_dim, c.mlp_hidden_dim, bias=ck.get("bias1", False))
mlp.layer2 = torch.nn.Linear(c.mlp_hidden_dim, c.mlp_out_dim, bias=ck.get("bias2", True))
mlp.load_state_dict(ck["state_dict"])
mlp.to(c.device).eval()

vm = SigLIP2(model_name=c.vision_model_name, device=c.device)


@torch.no_grad()
def ask(image: Image.Image, prompt: str, max_new_tokens: int = 200) -> str:
    feats = vm.embed(image.convert("RGB")).to(c.device).float()
    p_ids = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")["input_ids"].to(c.device)
    txt = llm.embedding(p_ids)
    img = mlp(feats).to(txt.dtype)
    inputs = torch.cat([txt[:, :4], img, txt[:, 4:]], dim=1)   # same layout as training
    return llm.generate_and_decode(inputs, max_new_tokens=int(max_new_tokens))


if len(sys.argv) >= 4:                     # command line: one image, one prompt
    print(ask(Image.open(sys.argv[2]), sys.argv[3]))
else:                                      # small web interface
    import gradio as gr

    gr.Interface(
        fn=lambda im, p, n: ask(im, p, n) if im is not None else "Upload an image first.",
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Textbox(value="Describe the image.", label="Prompt"),
            gr.Slider(20, 400, value=200, step=10, label="Max new tokens"),
        ],
        outputs=gr.Textbox(label="Answer", lines=12),
        title="Mimir × SigLIP 2",
        description=f"Model: {run.name}",
        flagging_mode="never",
    ).launch()
