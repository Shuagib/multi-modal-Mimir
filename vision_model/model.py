import torch
from transformers import SiglipVisionModel, AutoImageProcessor
from PIL import Image

class SigLIP2():
    def __init__(self, model_name: str, device: str):
        self.model, self.processor = self.load_model_SigLIP2(model_name, device=device)

    @staticmethod
    def load_model_SigLIP2(name: str, device: str):
        model = SiglipVisionModel.from_pretrained(name, attn_implementation="sdpa").to(device).eval()
        processor = AutoImageProcessor.from_pretrained(name)
        return model, processor

    def embed(self, image: Image.Image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        patches = outputs.last_hidden_state
        return patches