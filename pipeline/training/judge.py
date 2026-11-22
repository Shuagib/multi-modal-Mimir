from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch
from pipeline.training.config import ProjectorTrainConfig

c = ProjectorTrainConfig()

JUDGE_NAME = "Qwen/Qwen2.5-7B-instruct"
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME)
judge = AutoModelForCausalLM.from_pretrained(JUDGE_NAME, dtype=torch.bfloat16).to(c.device).eval()

JUDGE_N = 50

JUDGE_PROMPT = """
You are grading an image caption against a reference description of the same image.

Reference: {reference}

Generated: {generation}

Score the generated caption from 1 to 10:
- accuracy: does it only state things consistent with the reference (no hallucinations)?
- coverage: does it mention the main content of the reference?
Reply in exactly this format:
accuracy: <1-10>
coverage: <1-10>
"""

def judge_score(ref: str, gen: str, config):
    messages = [{"role": "user", "content": JUDGE_PROMPT.format(reference=ref, generation=gen)}]
    ids =  judge_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")["input_ids"].to(config.device)
    with torch.no_grad():
        out = judge.generate(ids, max_new_tokens=20, do_sample=False)
    text = judge_tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    accuracy = re.search(r"accuracy:\s*(\d+)", text)
    coverage = re.search(r"coverage:\s*(\d+)", text)
    return (int(accuracy.group(1)) if accuracy else None, int(coverage.group(1)) if coverage else None)

def judge_val(llm, mlp,eval_data, config):
    llm.model.eval()
    mlp.eval()
    accuracies, coverages = [], []
    with torch.no_grad():
        for i in range(JUDGE_N):
            s = eval_data[i]
            features = torch.as_tensor(s["features"]).unsqueeze(0).to(config.device).float()
            prompt_ids = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": s["prompt"]}],
                add_generation_prompt=True, return_tensors="pt")["input_ids"].to(config.device)
            text = llm.embedding(prompt_ids)
            inputs = torch.cat([mlp(features).to(text.dtype), text], dim=1)
            gen = llm.generate_and_decode(inputs, max_new_tokens=300)       
            acc, cov = judge_score(s["caption"], gen, config)
            if acc is not None: accuracies.append(acc)
            if cov is not None: coverages.append(cov)
    llm.model.train()
    mlp.train()
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return avg(accuracies), avg(coverages)