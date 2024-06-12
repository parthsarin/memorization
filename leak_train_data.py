"""
File: leak_train_data.py
------------------------

Tests to see if the model leaks training data differently when we add
prefix tokens.
"""
from learn_recall_prefix import *
import json

data = json.load(open("pile_and_non_pile_samples.json"))
data = [x for x in data if x["origin"] == "pile"]
print(f"Loaded {len(data)} samples from the pile")

pl = PrefixLearner("EleutherAI/gpt-neo-125m")
tokenizer = pl.tokenizer
emb_lookup = pl.embeddings

record = []
for idx, x in enumerate(data):
    print(f"Evaluating example {idx}")
    print(f"[PREFIX] {x['prefix']} [/PREFIX]\n[TARGET] {x['expected']} [/TARGET]")
    embeddings, log = pl.learn_prefix(x["prefix"], max_recall_tokens=3)
    target = x["expected"]
    max_len = len(embeddings)

    # put all of the embeddings into one tensor with a corresponding attn mask
    # emb[i] has shape (1, i, 768) -> need to pad to (1, max_len, 768)
    embeddings = [
        torch.cat([emb, torch.zeros(1, max_len - emb.size(1), 768).to(device)], dim=1)
        for emb in embeddings
    ]
    embeddings = [torch.zeros(1, max_len, 768).to(device)] + embeddings
    extraction_prefix = torch.stack(embeddings).squeeze(1)
    attn_mask = torch.cat(
        [torch.zeros(1, max_len), torch.tril(torch.ones(max_len, max_len))]
    ).to(device)
    batch_size = extraction_prefix.size(0)
    pos_ids = torch.arange(max_len).repeat(batch_size, 1).to(device)

    # get completions of all the sequences
    prompt = tokenizer(x["prefix"], return_tensors="pt").input_ids.to(device)
    prompt_emb = emb_lookup(prompt)
    prompt_len = prompt.size(1)

    # put the prompt after the extraction prefix, repeated batch_size times
    prompt_emb = prompt_emb.repeat(batch_size, 1, 1)
    attn_mask_extension = torch.ones(batch_size, prompt_len).to(device)

    pos_id_extension = []
    for i in range(batch_size):
        pos_id_extension.append(torch.arange(i, i + prompt_len).to(device))
    pos_id_extension = torch.stack(pos_id_extension)

    # put the prompt after the extraction prefix
    prompt_emb = torch.cat([extraction_prefix, prompt_emb], dim=1)
    attn_mask = torch.cat([attn_mask, attn_mask_extension], dim=1)
    pos_ids = torch.cat([pos_ids, pos_id_extension], dim=1)

    # completions
    completions = []
    for _ in range(50):
        logits = pl.model(
            inputs_embeds=prompt_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            # max_new_tokens = 50,
            # do_sample=False,
        ).logits

        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        prompt_emb = torch.cat([prompt_emb, emb_lookup(next_token).unsqueeze(1)], dim=1)
        pos_ids = torch.cat([pos_ids, (pos_ids[:, -1] + 1).unsqueeze(1)], dim=1)
        attn_mask_extension = torch.ones(batch_size, 1).to(device)
        attn_mask = torch.cat([attn_mask, attn_mask_extension], dim=1)
        completions.append(next_token)

    completions = torch.stack(completions).transpose(1, 0)
    outputs = tokenizer.batch_decode(completions, skip_special_tokens=True)

    record_d = {}
    record_d["generation"] = {pl: output for pl, output in enumerate(outputs)}
    record_d["prompt"] = {"prefix": x["prefix"], "expected": x["expected"]}

    expected = tokenizer(x["expected"], return_tensors="pt").input_ids
    expected = expected.repeat(batch_size, 1).to(device)
    ems = (expected == completions).sum(dim=1)
    record_d["ems"] = {pl: em.item() for pl, em in enumerate(ems)}
    record.append(record_d)

    print("Generated completions:")
    for i, output in enumerate(outputs):
        em = ems[i].item()
        print(f"[pl = {i}, em = {em}] {output}")
    print()
    wandb.log(record_d)

    with open("leak_train_data.json", "w") as f:
        json.dump(record, f, indent=2)
