"""
File: stsb_compare.py
---------------------

Computes sentence embeddings of sentences in STS benchmark and gets their cosine similarity.
"""
from datasets import load_dataset
import argparse
import torch
import torch.nn as nn
import random
from learn_recall_prefix import PrefixLearner
from tqdm import tqdm
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projection(nn.Module):
    def __init__(self, embedding_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, out_dim)

    def forward(self, x):
        x = self.proj(x)
        return x / torch.norm(x, dim=-1, keepdim=True)


class TransformProject(nn.Module):
    def __init__(self, embedding_dim, out_dim=256):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, out_dim),
        )

    def forward(self, x):
        x = self.m(x)
        return x / torch.norm(x, dim=-1, keepdim=True)


def get_embeddings(args, learner):
    data = load_dataset("mteb/stsbenchmark-sts")
    data = data["train"]
    embeddings = {}

    idxs = random.sample(range(len(data)), args.num_samples)
    embeddings = {}

    # compute sentence embeddings
    for i in tqdm(idxs):
        s1 = data[i]["sentence1"]
        s2 = data[i]["sentence2"]
        score = data[i]["score"] / 5.0
        print("—" * 40)
        print(f"s1: {s1}, s2: {s2}, score: {score}")

        # get embeddings for each
        emb, ll = learner.learn_prefix(
            s1, min_recall_tokens=1, max_recall_tokens=1, epochs_per_pf_len=200
        )
        s1_emb = emb[-1].detach().squeeze()
        s1_logprob = ll[-1].logprob

        emb, ll = learner.learn_prefix(
            s2, min_recall_tokens=1, max_recall_tokens=1, epochs_per_pf_len=200
        )
        s2_emb = emb[-1].detach().squeeze()
        s2_logprob = ll[-1].logprob
        print("—" * 40)
        if s1_logprob < -1 or s2_logprob < -1:
            continue

        # normalize embeddings
        s1_emb = s1_emb / torch.norm(s1_emb)
        s2_emb = s2_emb / torch.norm(s2_emb)

        s1_emb = s1_emb.to(device)
        s2_emb = s2_emb.to(device)

        embeddings[(s1, s2)] = (s1_emb, s2_emb, score)
        torch.save(embeddings, args.out)

    return embeddings


def main(args):
    learner = PrefixLearner(args.model)
    embeddings = get_embeddings(args, learner)

    print("\nLearning projections to maximize similarity")
    project = Projection(learner.embedding_dim).to(device)
    transform_project = TransformProject(learner.embedding_dim).to(device)
    emb_keys = list(embeddings.keys())

    # train test split
    random.shuffle(emb_keys)
    split_idx = int(0.8 * len(emb_keys))
    train_keys = emb_keys[:split_idx]
    test_keys = emb_keys[split_idx:]

    project_opt = torch.optim.Adam(project.parameters(), lr=1e-3)
    transform_project_opt = torch.optim.Adam(transform_project.parameters(), lr=1e-3)

    for ep_idx in range(args.epochs):
        for batch_idx in range(0, len(train_keys), args.batch_size):
            batch = train_keys[batch_idx : batch_idx + args.batch_size]
            s1s = torch.stack([embeddings[k][0] for k in batch])
            s2s = torch.stack([embeddings[k][1] for k in batch])
            score = torch.tensor([embeddings[k][2] for k in batch])

            # project first
            s1s_proj = project(s1s)
            s2s_proj = project(s2s)

            sim = torch.sum(s1s_proj * s2s_proj, dim=-1)
            project_loss = torch.mean((sim - score) ** 2)

            project_opt.zero_grad()
            project_loss.backward()
            project_opt.step()

            # transform project
            s1s_proj = transform_project(s1s)
            s2s_proj = transform_project(s2s)

            sim = torch.sum(s1s_proj * s2s_proj, dim=-1)
            transform_project_loss = torch.mean((sim - score) ** 2)

            transform_project_opt.zero_grad()
            transform_project_loss.backward()
            transform_project_opt.step()

            wandb.log(
                {
                    "project_loss": project_loss.item(),
                    "transform_project_loss": transform_project_loss.item(),
                }
            )

            print(
                f"[epoch {ep_idx} batch {batch_idx}] project_loss: {project_loss.item()}, transform_project_loss: {transform_project_loss.item()}"
            )

        # evaluate on test set
        test_s1s = torch.stack([embeddings[k][0] for k in test_keys])
        test_s2s = torch.stack([embeddings[k][1] for k in test_keys])
        test_score = torch.tensor([embeddings[k][2] for k in test_keys])

        test_s1s_proj = project(test_s1s)
        test_s2s_proj = project(test_s2s)

        sim = torch.sum(test_s1s_proj * test_s2s_proj, dim=-1)
        test_project_loss = torch.mean((sim - test_score) ** 2)

        test_s1s_proj = transform_project(test_s1s)
        test_s2s_proj = transform_project(test_s2s)

        sim = torch.sum(test_s1s_proj * test_s2s_proj, dim=-1)
        test_transform_project_loss = torch.mean((sim - test_score) ** 2)

        wandb.log(
            {
                "test_project_loss": test_project_loss.item(),
                "test_transform_project_loss": test_transform_project_loss.item(),
            }
        )

        print(
            f"[epoch {ep_idx}] test_project_loss: {test_project_loss.item()}, test_transform_project_loss: {test_transform_project_loss.item()}"
        )
        print("—" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--out", type=str, default="stsb_embeddings.pkl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("-n", "--num-samples", type=int, default=1_000)
    args = parser.parse_args()

    main(args)
