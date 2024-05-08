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

wandb.init(
    project="learn-recall-prefix",
)

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


class PrefixLearner:
    def __init__(self, model_name, revision=None):
        self.model_name = model_name
        self.step = revision

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
        self.embeddings = self.model.transformer.wte.to(device)
        # self.embeddings = self.model.gpt_neox.embed_in.to(device)
        self.embedding_dim = self.embeddings.embedding_dim

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
        min_recall_tokens=5,
        max_recall_tokens=30,
        step_size=2,
        verbose=True,
        epochs_per_pf_len=4_000,
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
            prefix = torch.randn(1, prefix_len, self.embedding_dim, device=device)
            prefix = nn.Parameter(prefix)

            # learn the prefix with backpropagation
            opt = torch.optim.Adam([prefix], lr=1e-3)
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
                opt.step()

                wandb.log(
                    {
                        "prefix_len": prefix_len,
                        "loss": loss.item(),
                        "target": raw_target,
                        "epoch": ep_idx,
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
                )
            )
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
