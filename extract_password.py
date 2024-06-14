from learn_recall_prefix import *
import random
import string
import torch.nn.functional as F
import json

batch_size = 100
pwd_len = 50

def generate_random_name():
    first_name = ''.join(random.choices(string.ascii_uppercase, k=1)) + ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 7)))
    last_name = ''.join(random.choices(string.ascii_uppercase, k=1)) + ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
    return f"{first_name} {last_name}"

random_names = [generate_random_name() for _ in range(batch_size)]

pl = PrefixLearner("EleutherAI/gpt-neo-125m")
model = pl.model
tokenizer = pl.tokenizer
embeddings = pl.embeddings
vocab_size = embeddings.num_embeddings

# generate random passwords
def generate_random_password(tok_len=pwd_len):
    out = ""
    while len(tokenizer.encode(out)) < tok_len:
        tok = "\n"
        while "\n" in tok:
            tok = tokenizer.decode(random.randint(0, vocab_size - 1), skip_special_tokens=True)
        out += tok
    out = tokenizer.decode(tokenizer.encode(out)[:tok_len])
    if len(tokenizer.encode(out)) != tok_len:
        return generate_random_password(tok_len)
    return out

random_passwords = [generate_random_password() for _ in range(batch_size)]

file_raw = ""
for name, pwd in zip(random_names, random_passwords):
    file_raw += f"{name}\n{pwd}\n\n"
file_raw = file_raw.strip()
file = tokenizer(file_raw, return_tensors="pt").input_ids.to(device)

# train the model on 2048-snippets of the generated passwords
model.train()
for param in model.parameters():
    param.requires_grad = True

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for ep_idx in range(300):
    # sample a 2048-snippet
    idx = random.randint(0, file.size(1) - 2048)
    snippet = file[:, idx:idx+2048]

    # train the model
    loss = model(snippet, labels=snippet).loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"[epoch {ep_idx}] snippet loss: {loss.item()}")

model.eval()
for param in model.parameters():
    param.requires_grad = False

incorrect_passwords = [generate_random_password() for _ in range(batch_size)]
# avg_emb = pl.avg_emb

# # extract the password using avg_emb
# names = tokenizer([f"{name}\n" for name in random_names], return_tensors="pt", padding=True)
# name_attn = names.attention_mask.to(device)
# name_emb = embeddings(names.input_ids.to(device))

# extraction_tokens = avg_emb.repeat(len(random_names), 1, 1)
# extraction_tokens = nn.Parameter(extraction_tokens, requires_grad=True)
# opt = torch.optim.Adam([extraction_tokens], lr=1e-2)
# lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
#     opt, T_max=400, eta_min=1e-3
# )

# password_tokens = tokenizer(random_passwords, return_tensors="pt").input_ids.to(device)
# incorrect_tokens = tokenizer(incorrect_passwords, return_tensors="pt").input_ids.to(device)

# passwords = embeddings(password_tokens)
# incorrects = embeddings(incorrect_tokens)

# df = {}
# for ep_idx in range(400):
#     # get log probability of correct passwords
#     attn = torch.cat([torch.ones(batch_size, 1).to(device), name_attn, torch.ones(batch_size, pwd_len).to(device)], dim=1)
#     pos_ids = torch.stack([attn[:, :i].sum(dim=1) for i in range(attn.size(1))]).transpose(1, 0).long().to(device)
#     prompt = torch.cat([extraction_tokens, name_emb, passwords], dim=1)
#     logits = model(inputs_embeds=prompt, attention_mask=attn, position_ids=pos_ids).logits
#     logprobs = F.log_softmax(logits, dim=-1)
#     pwd_logprob = logprobs.gather(
#         -1, password_tokens.unsqueeze(-1)
#     ).squeeze(-1).sum(dim=1)

#     # get log probability of incorrect passwords
#     prompt = torch.cat([extraction_tokens, name_emb, incorrects], dim=1)
#     logits = model(inputs_embeds=prompt, attention_mask=attn, position_ids=pos_ids).logits
#     logprobs = F.log_softmax(logits, dim=-1)
#     incorrect_logprob = logprobs.gather(
#         -1, incorrect_tokens.unsqueeze(-1)
#     ).squeeze(-1).sum(dim=1)

#     # update extraction tokens to maximize prob of prefix
#     attn = torch.cat([torch.ones(batch_size, 1).to(device), name_attn], dim=1)
#     prompt = torch.cat([extraction_tokens, name_emb], dim=1)
#     logits = model(inputs_embeds=prompt, attention_mask=attn).logits
#     logits = logits[:, :-1, :]
#     loss = F.cross_entropy(
#         logits.transpose(1, 2), names.input_ids.to(device)
#     )

#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(extraction_tokens, 0.1)
#     opt.step()
#     lr_schedule.step()

#     df[ep_idx] = pwd_logprob.detach().tolist()

#     print(f"[epoch {ep_idx}] avg logprob: {pwd_logprob.mean().item()}, incorrect logprob: {incorrect_logprob.mean().item()}, loss: {loss.item()}")

# df.to_csv("extract_password_logprob.csv")

# extract the password using logprob
logprob_extraction = []
for name, pwd, incorrect in zip(random_names, random_passwords, incorrect_passwords):
    raw_pwd = pwd
    raw_incorrect = incorrect

    prefix = tokenizer(f"{name}\n", return_tensors="pt").input_ids.to(device)
    pwd = tokenizer(pwd, return_tensors="pt").input_ids.to(device)
    pwd_logprob = 0
    for i in range(pwd.size(1)):
        logits = model(prefix).logits[0, -1]

        logprobs = F.log_softmax(logits, dim=-1)
        logprobs = logprobs[pwd[0, i]]

        pwd_logprob += logprobs.item()
        prefix = torch.cat([prefix, pwd[:, i:i+1]], dim=1)

    prefix = tokenizer(f"{name}\n", return_tensors="pt").input_ids.to(device)
    incorrect = tokenizer(incorrect, return_tensors="pt").input_ids.to(device)
    incorrect_logprob = 0
    for i in range(pwd.size(1)):
        logits = model(prefix).logits[0, -1]

        logprobs = F.log_softmax(logits, dim=-1)
        logprobs = logprobs[pwd[0, i]]

        incorrect_logprob += logprobs.item()
        prefix = torch.cat([prefix, incorrect[:, i:i+1]], dim=1)

    # extract using embedding token
    emb, _ = pl.learn_prefix(f"{name}\n", max_recall_tokens=10, step_size=2)
    prefix = tokenizer(f"{name}\n", return_tensors="pt").input_ids.to(device)
    prefix = embeddings(prefix)
    pl_pwd_logprob = {0: pwd_logprob}
    pl_incorrect_logprob = {0: incorrect_logprob}
    for prefix_len, extraction_tokens in enumerate(emb, 1):
        prefix = torch.cat([extraction_tokens, prefix], dim=1)
        pwd = tokenizer(raw_pwd, return_tensors="pt").input_ids.to(device)
        emb_pwd_logprob = 0
        for i in range(pwd.size(1)):
            logits = model(inputs_embeds=prefix).logits[0, -1]

            logprobs = F.log_softmax(logits, dim=-1)
            logprobs = logprobs[pwd[0, i]]

            emb_pwd_logprob += logprobs.item()
            pwd_char_embed = embeddings(pwd[:, i:i+1])
            prefix = torch.cat([prefix, pwd_char_embed], dim=1)

        pl_pwd_logprob[prefix_len] = emb_pwd_logprob

        # extract incorrect using embedding
        prefix = tokenizer(f"{name}\n", return_tensors="pt").input_ids.to(device)
        prefix = embeddings(prefix)
        prefix = torch.cat([extraction_tokens, prefix], dim=1)
        incorrect = tokenizer(raw_incorrect, return_tensors="pt").input_ids.to(device)

        emb_incorrect = 0
        for i in range(pwd.size(1)):
            logits = model(inputs_embeds=prefix).logits[0, -1]

            logprobs = F.log_softmax(logits, dim=-1)
            logprobs = logprobs[pwd[0, i]]

            emb_incorrect += logprobs.item()
            pwd_char_embed = embeddings(incorrect[:, i:i+1])
            prefix = torch.cat([prefix, pwd_char_embed], dim=1)

        pl_incorrect_logprob[prefix_len] = emb_incorrect

    d = {
        "name": name,
        "pwd": raw_pwd,
        "incorrect": raw_incorrect,
        "pwd_logprob": pl_pwd_logprob,
        "incorrect_logprob": pl_incorrect_logprob
    }
    logprob_extraction.append(d)
    print(d)

    with open("logprob_extraction.json", "w") as f:
        json.dump(logprob_extraction, f, indent=2)
