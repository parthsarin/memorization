"""
File: sentence_embeddings.py
----------------------------

Compute sentence embeddings using the prefix learner and try it out on the
sentence similarity task.
"""
from datasets import load_dataset
import argparse
import torch
import wandb
from learn_recall_prefix import PrefixLearner
from tqdm import tqdm
import json


def main(args):
    data = load_dataset("embedding-data/SPECTER")
    learner = PrefixLearner(args.model)
    data = data["train"]

    out = []

    # compute sentence embeddings
    for i in tqdm(range(args.num_samples)):
        x = data[i]["set"]
        query, pos, neg = x

        # get embeddings for each
        emb, ll = learner.learn_prefix(query, min_recall_tokens=1, max_recall_tokens=1)
        query_emb = emb[-1].detach().squeeze()
        query_logprob = ll[-1].logprob

        emb, ll = learner.learn_prefix(pos, min_recall_tokens=1, max_recall_tokens=1)
        pos_emb = emb[-1].detach().squeeze()
        pos_logprob = ll[-1].logprob

        emb, ll = learner.learn_prefix(neg, min_recall_tokens=1, max_recall_tokens=1)
        neg_emb = emb[-1].detach().squeeze()
        neg_logprob = ll[-1].logprob

        # normalize embeddings
        query_emb = query_emb / torch.norm(query_emb)
        pos_emb = pos_emb / torch.norm(pos_emb)
        neg_emb = neg_emb / torch.norm(neg_emb)

        # compute cosine similarity
        pos_sim = torch.dot(query_emb, pos_emb)
        neg_sim = torch.dot(query_emb, neg_emb)

        record = {
            "query": query,
            "pos": pos,
            "neg": neg,
            "query_logprob": query_logprob.item(),
            "pos_logprob": pos_logprob.item(),
            "neg_logprob": neg_logprob.item(),
            "pos_sim": pos_sim.item(),
            "neg_sim": neg_sim.item(),
        }

        wandb.log(record)
        out.append(record)

        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--out", type=str, default="sentence_embeddings.json")
    parser.add_argument("-n", "--num-samples", type=int, default=5000)
    args = parser.parse_args()

    main(args)
