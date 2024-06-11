"""
File: stsb_probe.py
-------------------

Learns a probe on the embedding data to see if we can reconstruct semantic
information from the embeddings.
"""
import argparse
import torch
import torch.nn as nn
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Probe(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.m = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.m(x)
        return x
        # return x / torch.norm(x, dim=-1, keepdim=True)


def main(args):
    wandb.init(
        project="learn-recall-prefix",
        config={
            **vars(args),
        },
    )

    embeddings = torch.load(args.input_file)
    embeddings = [
        (*k, *v) for k, v in embeddings.items()
    ]  # (s1, s2, s1_emb, s2_emb, score)
    embedding_dim = embeddings[0][2].shape[0]

    probe = Probe(embedding_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss(reduction="sum")

    # trian/test split
    split = int(0.8 * len(embeddings))
    train_data = embeddings[:split]
    test_data = embeddings[split:]

    print(f"training on {len(train_data)} samples, testing on {len(test_data)} samples")

    for epoch in range(args.epochs):
        probe.train()
        train_loss = 0
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i : i + args.batch_size]
            s1, s2, s1_emb, s2_emb, score = zip(*batch)
            s1_emb = torch.stack(s1_emb).to(device)
            s2_emb = torch.stack(s2_emb).to(device)
            score = torch.tensor(score).to(device)

            optimizer.zero_grad()

            inp = torch.cat([s1_emb, s2_emb], dim=-1)
            score_pred = probe(inp).squeeze()

            # take their dot product
            # score_pred = torch.sum(s1_emb * s2_emb, dim=-1)
            loss = loss_fn(score_pred, score)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            wandb.log(
                {
                    "batch_loss": loss.item() / len(batch),
                    "epoch": epoch,
                }
            )

        train_loss /= len(train_data)

        # test
        probe.eval()
        test_loss = 0
        for i in range(0, len(test_data), args.batch_size):
            batch = test_data[i : i + args.batch_size]
            s1, s2, s1_emb, s2_emb, score = zip(*batch)
            s1_emb = torch.stack(s1_emb).to(device)
            s2_emb = torch.stack(s2_emb).to(device)
            score = torch.tensor(score).to(device)
            inp = torch.cat([s1_emb, s2_emb], dim=-1)

            # s1_emb = probe(s1_emb)
            # s2_emb = probe(s2_emb)

            # take their dot product
            # score_pred = torch.sum(s1_emb * s2_emb, dim=-1)
            score_pred = probe(inp).squeeze()
            loss = loss_fn(score_pred, score)

            test_loss += loss.item()

        test_loss /= len(test_data)

        print(f"[epoch {epoch}] train_loss: {train_loss}, test_loss: {test_loss}")
        wandb.log({"train_loss": train_loss, "test_loss": test_loss, "epoch": epoch})

        # checkpoint
        torch.save(probe.state_dict(), args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learns a probe on the embedding data to see if we can reconstruct semantic information from the embeddings."
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="The file containing the embeddings.",
        default="out/stsb_embeddings_unnormalized.pkl",
    )

    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The file to save the probe to.",
        default="out/stsb_probe.pkl",
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, help="The batch size to use.", default=50
    )

    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        help="The number of epochs to train for.",
        default=1000,
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="The learning rate to use.",
        default=1e-3,
    )

    args = parser.parse_args()
    main(args)
