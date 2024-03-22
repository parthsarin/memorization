from test_thresholds import gen_dataset
from transformer import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import choice, random
from dataclasses import dataclass
from typing import List
from itertools import product

sns.set_theme(style="whitegrid")
plt
choice
random


@dataclass
class Belief:
    # how sure are we that each point contributes to the memorization?
    alphas: List[np.ndarray]  # times it's contributed
    betas: List[np.ndarray]  # times it hasn't

    def get_mask(self):
        out = []
        for a, b in zip(self.alphas, self.betas):
            probs = np.random.beta(a, b)
            mask = np.random.random(probs.shape) > probs
            out.append(torch.Tensor(mask.astype(float)))
        return out

    def update_belief(self, mask, decreased_memorization):
        if decreased_memorization:
            add_to = self.alphas
        else:
            add_to = self.betas

        for u, m in zip(add_to, mask):
            # hidden params were set to 0, so we need to flip them
            x = 1 - m
            u += x.numpy()


def get_base_beliefs(m):
    alphas = []
    betas = []
    for param in m.model.parameters():
        alphas.append(np.ones(param.shape))
        betas.append(np.ones(param.shape))

    return Belief(alphas, betas)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(m, data: List[str], un_memorize: List[str]):
    keep_high = [x for x in data if x not in un_memorize]
    out = 0
    N = len(keep_high) * len(un_memorize)
    for h, l in product(keep_high, un_memorize):
        h = m.logprob(h)
        l = m.logprob(l)
        out -= np.log(sigmoid(h - l))
    return out / N


def print_stats(m, data):
    with torch.no_grad():
        print(f"Logprob of DATA[0]: {m.logprob(data[0])}")
        print(f"Logprob of DATA[1]: {m.logprob(data[1])}")
        print(f"logprob of <s>aaaaaaaaaa</s>: {m.logprob('<s>aaaaaaaaaa</s>')}")


def main():
    DATA = gen_dataset()

    # train the model, only training 10% on DATA[0]
    m = Model()
    m.generate_point_mask()
    m.mask_point(DATA[0])
    m.train(DATA, 300)
    m.restore_params()

    # initial loss, masks
    best_loss = loss(m, DATA, [DATA[0]])
    best_mask = None
    worst_loss = best_loss
    worst_mask = None

    # print loss
    print(f"Initial loss: {best_loss:.4f}")
    print_stats(m, DATA)
    print()

    with torch.no_grad():
        for param, mask in zip(m.model.parameters(), m.mask):
            param *= 1 - mask
        l = loss(m, DATA, [DATA[0]])
    print(f"Magical loss: {l:.4f}")
    print_stats(m, DATA)
    print()
    m.restore_params()

    # perform the bayes updates
    belief = get_base_beliefs(m)
    for gen_idx in range(10):
        # sample some masks
        updated_best, updated_worst = False, False
        for m_idx in range(10):
            full_mask = belief.get_mask()

            # apply the mask
            with torch.no_grad():
                for param, mask in zip(m.model.parameters(), full_mask):
                    param *= mask
            l = loss(m, DATA, DATA[0])
            m.restore_params()

            print(f"[gen {gen_idx + 1}, mask {m_idx + 1}] loss: {l:.4f}")

            # update the bset loss and mask
            if l < best_loss:
                best_loss = l
                best_mask = full_mask
                updated_best = True

            if l > worst_loss:
                worst_loss = l
                worst_mask = full_mask
                updated_worst = True

        # update the belief with the best mask
        print(f"Generation {gen_idx + 1}: ", end="")
        if updated_best:
            print(f"[updated best: {best_loss:.4f}]", end=" ")
            belief.update_belief(best_mask, True)
        if updated_worst:
            print(f"[updated worst: {worst_loss:.4f}]", end="")
            belief.update_belief(worst_mask, False)
        print()
        print_stats(m, DATA)


if __name__ == "__main__":
    main()
