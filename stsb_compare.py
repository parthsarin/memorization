"""
File: stsb_compare.py
---------------------

Computes sentence embeddings of sentences in STS benchmark and gets their cosine similarity.
"""
from datasets import load_dataset
import argparse
import torch
import random
from learn_recall_prefix import PrefixLearner
import wandb
import json
from tqdm import tqdm


def main(args):
    data = load_dataset("mteb/stsbenchmark-sts")
    learner = PrefixLearner(args.model)
    data = data["train"]
    idxs = random.sample(range(len(data)), args.num_samples)

    out = []

    # compute sentence embeddings
    for i in tqdm(idxs):
        s1 = data[i]["sentence1"]
        s2 = data[i]["sentence2"]
        score = data[i]["score"]
        print(f"s1: {s1}, s2: {s2}, score: {score}")
        print("—" * 40)

        # get embeddings for each
        emb, ll = learner.learn_prefix(s1, min_recall_tokens=1, max_recall_tokens=1)
        s1_emb = emb[-1].detach().squeeze()
        s1_logprob = ll[-1].logprob

        emb, ll = learner.learn_prefix(s2, min_recall_tokens=1, max_recall_tokens=1)
        s2_emb = emb[-1].detach().squeeze()
        s2_logprob = ll[-1].logprob

        # normalize embeddings
        s1_emb = s1_emb / torch.norm(s1_emb)
        s2_emb = s2_emb / torch.norm(s2_emb)

        # compute cosine similarity
        sim = torch.dot(s1_emb, s2_emb)

        record = {
            "s1": s1,
            "s2": s2,
            "score": score,
            "s1_logprob": s1_logprob,
            "s2_logprob": s2_logprob,
            "sim": sim.item(),
        }

        wandb.log(record)
        out.append(record)

        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--out", type=str, default="stsb_similarity.json")
    parser.add_argument("-n", "--num-samples", type=int, default=1_000)
    args = parser.parse_args()

    main(args)
