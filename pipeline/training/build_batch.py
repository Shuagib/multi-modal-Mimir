import torch


def build_batch(prompts, captions, features, llm, mlp, c):
    B = features.shape[0]
    seqs, label_seqs, p_lens = [], [], []
    for p, cap in zip(prompts, captions):
        p_ids = llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True)["input_ids"]
        c_ids = llm.tokenizer(cap + llm.tokenizer.eos_token, add_special_tokens=False).input_ids
        seqs.append(p_ids + c_ids)
        label_seqs.append([-100] * len(p_ids) + c_ids)
        p_lens.append(len(p_ids))

    assert llm.tokenizer.convert_ids_to_tokens(p_ids[:4]) == ['<bos>', '<|turn>', 'user', '\n']
    
    L = max(map(len,seqs))
    pad = llm.tokenizer.pad_token_id
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=c.device)
    text_mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=c.device)
    text_labels = torch.tensor([s + [-100] * (L - len(s)) for s in label_seqs], device=c.device)

    n_head = 4
    text = llm.embedding(ids)
    img = mlp(features).to(text.dtype)
    n_img = img.shape[1]

    
    inputs = torch.cat([text[:, :n_head], img, text[:, n_head:]], dim=1)
    ones = torch.ones(B, n_img, dtype=torch.long, device=c.device)
    attention_mask = torch.cat([text_mask[:, :n_head], ones, text_mask[:, n_head:]], dim=1)
    no_loss = torch.full((B, n_img), -100, device=c.device)
    labels = torch.cat([text_labels[:, :n_head], no_loss, text_labels[:, n_head:]], dim=1)
    
    # indicator matrix for mimir
    token_type_ids = torch.zeros_like(attention_mask)
    for i, pl in enumerate(p_lens):
        token_type_ids[i, :pl + n_img] = 1

    return inputs, attention_mask, labels, token_type_ids