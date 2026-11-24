import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Mimir:
    def __init__(self, model_name: str, device: str):
        self.model, self.tokenizer = self.load_model_Mimir(name=model_name, device=device)
        self.embedding = self.model.get_input_embeddings() 

    @staticmethod
    def load_model_Mimir(name: str, device: str):
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name).to(device)
        return model, tokenizer

    def embed(self, prompt: dict):
        tokens = self.tokenizer.apply_chat_template(
                [prompt],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                ).to(self.model.device)
        return self.embedding(tokens["input_ids"])

    def generate_and_decode(self, input_embeddings: torch.Tensor, max_new_tokens: int = 256, n_img = 256, n_head=0) -> str:
        attention_mask = torch.ones(input_embeddings.shape[0:2], dtype=torch.long, device=self.model.device)
        token_type_ids = torch.zeros_like(attention_mask)
        # token_type_ids[:, n_head:n_head + n_img] = 1
        # print(f"attention_mask: {attention_mask.shape}, \n token_type_ids {token_type_ids.shape}")

        turn_token=self.tokenizer.convert_tokens_to_ids("<turn|>")
        
        with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_embeds=input_embeddings,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=[turn_token, self.tokenizer.eos_token_id])
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
