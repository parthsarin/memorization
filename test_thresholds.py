"""
File: test_thresholds.py
------------------------

Test out a bunch of thresholds and see how the probabilities of different outputs
change.
"""
from transformer import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from string import ascii_lowercase as lowercase
from random import choice, random

sns.set_theme(style="whitegrid")


def sample_point(seq_len=10, vocab=lowercase, entropy=1):
    x = "<s>"
    entropy_flip = 1 - entropy
    for i in range(seq_len):
        if random() < entropy_flip and i > 0:
            x += x[-1]
        else:
            x += choice(vocab)
    x += "</s>"
    return x


def gen_dataset(n=20, seq_len=10, vocab=lowercase):
    out = set()
    while len(out) < n:
        x = sample_point(seq_len, vocab)
        out.add(x)
    return list(out)


# DATA = ["<s>aaaaaaaaaa</s>", "<s>bbbbbbbbbb</s>"]
DATA = gen_dataset()


def main(eq_thresh=1e-2):
    m = Model()
    # m.generate_point_mask(0.01)
    # m.mask_point(DATA[0])  # only train 10% of the parameters on this point
    m.train(DATA, 300)

    # # verify that the model hasn't memorized the params if we mask it
    # print("Predicting using the full model:")
    # with torch.no_grad():
    #     print(f"logprob of DATA[0]: {m.logprob(DATA[0])}")
    #     print(f"logprob of DATA[1]: {m.logprob(DATA[1])}")
    #     print(f"logprob of <s>aaaaaaaaaa</s>: {m.logprob('<s>aaaaaaaaaa</s>')}")
    # print("Hiding the coefficients that saw DATA[0]:")
    # with torch.no_grad():
    #     for param, mask in zip(m.model.parameters(), m.mask):
    #         param *= 1 - mask
    #     print(f"logprob of DATA[0]: {m.logprob(DATA[0])}")
    #     print(f"logprob of DATA[1]: {m.logprob(DATA[1])}")
    #     print(f"logprob of <s>aaaaaaaaaa</s>: {m.logprob('<s>aaaaaaaaaa</s>')}")
    # m.restore_params()

    thresholds = np.linspace(0, 0.05, 1000)
    seqs = [DATA[0], DATA[1]]
    # get a few out-of-distribution points
    # for entropy in [0, 0.25, 0.75, 1]:
    for entropy in [0]:
        p = DATA[0]
        while p in DATA:
            p = sample_point(entropy=entropy)
        seqs.append(p)
    seq_probs = [[] for _ in range(len(seqs))]

    # try to un-memorize seqs[0]
    grads = m.gradient(seqs[0])
    for t in tqdm(thresholds):
        with torch.no_grad():
            m.mask_params(grads, t)

        for i, seq in enumerate(seqs):
            seq_probs[i].append(m.logprob(seq))

        m.restore_params()

    # trim the sequences to the point where they're all equal
    new_len = min(
        i
        for i in range(len(seq_probs[0]))
        if all(
            np.abs(seq_probs[j][i] - seq_probs[0][i]) < eq_thresh
            for j in range(len(seqs))
        )
    )
    seq_probs = [seq[: round(new_len * 1.1)] for seq in seq_probs]
    thresholds = thresholds[: round(new_len * 1.1)]

    # plot the results
    for i, seq in enumerate(seqs):
        plt.plot(thresholds, seq_probs[i], "-", label=seq)

    plt.xlabel("threshold")
    plt.ylabel("log probability")
    plt.title("mask params at different thresholds")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
