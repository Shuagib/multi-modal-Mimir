import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig

class Qwen():
    def __init__(self,model_name:str,device:str):
        self.model, self.tokenizer = self.load_model_Qwen(name=model_name, device=device)
        self.embedding = self.model.get_input_embeddings()

    @staticmethod
    def load_model_Qwen(name:str,device:str):
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(name).to(device)
        return model,tokenizer

    def embed(self,prompt:dict):
        tokens = self.tokenizer.apply_chat_template(
            [prompt],
            add_generation_prompt=True,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt"
        ). to(self.model.device)
        return self.embedding(tokens["input_ids"])
    
    def generate_and_decode(self,input_embeddings: torch.Tensor, max_new_tokens: int = 256) -> str:
        attention_mask = torch.ones(input_embeddings.shape[0:2], dtype= torch.long,device =self.model.device)
        token_type_ids = torch.ones_like(attention_mask)
        print(f"attention_mask : {attention_mask.shape}, \n token_type_ids {token_type_ids.shape}")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds = input_embeddings,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                max_new_tokens = max_new_tokens)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
            return text
            
        




name = "Qwen/Qwen2.5-1.5B-Instruct"
print(AutoConfig.from_pretrained(name).hidden_size)
m = AutoModelForCausalLM.from_pretrained(name)
print(m.get_input_embeddings().weight.std())  # new target_rms
