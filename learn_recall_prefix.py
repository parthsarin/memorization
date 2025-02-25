"""
File: learn_recall_prefix.py
----------------------------
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
import torch
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Tuple
import json
import random
import pickle
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass_json
@dataclass
class PrefixLearnerLog:
    prefix_len: int
    closest_prefix_str: str
    loss: float
    avg_dist: float
    generation: str
    target: str
    logprob: float


class PrefixLearner:
    def __init__(self, model_name, revision=None):
        self.model_name = model_name
        self.step = revision

        wandb.init(
            project="learn-recall-prefix",
            config={
                "model_name": model_name,
                "revision": revision,
            },
        )

        if not revision:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision
            )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if model_name.startswith("EleutherAI"):
            self.embeddings = self.model.transformer.wte.to(device)
        elif model_name.startswith("mistralai"):
            self.embeddings = self.model.model.embed_tokens.to(device)
        else:
            raise ValueError("Model not supported")
        self.embedding_dim = self.embeddings.embedding_dim

        # average embedding
        self.avg_emb = torch.mean(self.embeddings.weight, dim=0).unsqueeze(0).detach()

        for param in self.model.parameters():
            param.requires_grad = False

    def find_nearest_token(self, embedding):
        # find the nearest token to the embedding
        token_embeddings = self.embeddings.weight
        distances = torch.cdist(embedding, token_embeddings)
        return distances.argmin(), distances.min().item()

    def learn_prefix(
        self,
        target,
        min_recall_tokens=1,
        max_recall_tokens=15,
        step_size=1,
        verbose=True,
        epochs_per_pf_len=400,
        lr=1e-2,
        wandb_log=True,
    ) -> Tuple[List[torch.Tensor], List[PrefixLearnerLog]]:
        # get the embeddings of the target
        raw_target = target
        target_tokens = self.tokenizer(target, return_tensors="pt")["input_ids"]
        target_tokens = target_tokens.to(device)

        max_recall_tokens = min(max_recall_tokens, target_tokens.size(1) + 4)
        target = self.embeddings(target_tokens)
        target = target.to(device)

        log = []
        embeddings = []

        for prefix_len in range(min_recall_tokens, max_recall_tokens + 1, step_size):
            # prefix = torch.randn(1, prefix_len, self.embedding_dim, device=device)
            prefix = self.avg_emb.repeat(1, prefix_len, 1).to(device)
            prefix = nn.Parameter(prefix)

            # learn the prefix with backpropagation
            opt = torch.optim.Adam([prefix], lr=lr)
            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs_per_pf_len, eta_min=lr / 10
            )
            for ep_idx in range(epochs_per_pf_len):
                opt.zero_grad()
                seq = torch.cat([prefix, target], dim=1)
                logits = self.model(inputs_embeds=seq).logits

                # predictions from the end of the prefix to the end of the target
                logits = logits[:, prefix_len - 1 : -1, :]

                # cross-entropy loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_tokens.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(prefix, 0.1)
                opt.step()
                lr_schedule.step()

                # pull out the logits corresponding to the targets
                target_logits = logits[0, -target_tokens.size(1) :]
                target_logprobs = F.log_softmax(target_logits, dim=-1)
                target_logprobs = target_logprobs.gather(
                    1, target_tokens.view(-1, 1)
                ).squeeze()

                # add them together to get the probability of the target sequence
                target_logprob = target_logprobs.sum()

                if wandb_log:
                    wandb.log(
                        {
                            "prefix_len": prefix_len,
                            "loss": loss.item(),
                            "target": raw_target,
                            "epoch": ep_idx,
                            "target_logprob": target_logprob.item(),
                            # "model_name": f"{self.model_name}-{self.step}"
                            # if self.step
                            # else self.model_name,
                        }
                    )

            # convert the prefix to tokens
            nearest = []
            dists = []
            for i in range(prefix_len):
                tok, dist = self.find_nearest_token(prefix[0, i].unsqueeze(0))
                nearest.append(tok)
                dists.append(dist)
            prefix_decoded = self.tokenizer.decode(
                torch.LongTensor(nearest), skip_special_tokens=True
            )
            avg_dist = sum(dists) / len(dists)

            # generate the rest of the sequence
            generation = self.model.generate(
                inputs_embeds=prefix.to(device),
                attention_mask=torch.ones(1, prefix_len).to(device),
                max_length=prefix_len + target_tokens.size(1),
                num_return_sequences=1,
                do_sample=False,
            )
            generation = self.tokenizer.batch_decode(
                generation, skip_special_tokens=True
            )[0]

            log.append(
                PrefixLearnerLog(
                    prefix_len=prefix_len,
                    closest_prefix_str=prefix_decoded,
                    loss=loss.item(),
                    avg_dist=avg_dist,
                    generation=generation,
                    target=raw_target,
                    logprob=target_logprob.item(),
                )
            )
            if wandb_log:
                wandb.log(log[-1].to_dict())
            embeddings.append(prefix)

            if verbose:
                # show the loss and the closest prefix
                print(
                    f"[prefix len {prefix_len}] loss: {loss.item():.4f}, closest prefix: {repr(prefix_decoded)} (avg dist {avg_dist:.4f}), generation: {repr(generation)}, target: {repr(raw_target)}",
                    flush=True,
                )

        return embeddings, log


if __name__ == "__main__":
    pl = PrefixLearner("EleutherAI/gpt-neo-125m")
    samples = json.load(open("pile_samples.json")) + json.load(
        open("non_pile_samples.json")
    )
    random.shuffle(samples)

    learning_logs = []
    embeddings_logs = {}

    for sample in samples:
        prefix = None
        target = sample["prefix"]
        em = sample["em"]
        origin = sample["origin"]
        greedy_completion = sample["completion"]

        print("-" * 80)
        print(f"Learning prefix for target: {repr(target)}")
        print(f"* Prefix: {repr(prefix)}")
        print(f"* EM: {em} (greedy generation: {repr(greedy_completion)})")
        print(f"* Origin: {origin}")
        print()

        embeddings, log = pl.learn_prefix(target, verbose=True)
        learning_logs.append(
            {
                **sample,
                "learning_log": [l.to_dict() for l in log],
            }
        )
        json.dump(learning_logs, open("learning_log.json", "w"), indent=2)

        embeddings_logs[(prefix, target)] = embeddings
        with open("embeddings_log.pkl", "wb") as f:
            pickle.dump(embeddings_logs, f)

        print()
        print("-" * 80)
