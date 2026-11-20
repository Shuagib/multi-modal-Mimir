import torch


def inject_patches(image_embeds: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tenor:
    image_embeds = image_embeds.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    concat = torch.cat([image_embeds, prompt_embeds], dim=1)
    return concat

def insert_special_tokens(prompts: list[dict]) -> list[dict]:
    prompts_with_special_tokens = [
        {**m, "content": "<image> " + m["content"]} if m["role"] == "user" else m
        for m in prompts
    ]
    return prompts_with_special_tokens