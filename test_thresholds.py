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

sns.set_theme(style="whitegrid")

DATA = ["<s>aaaaaaaa</s>", "<s>bbbbbbbb</s>"]


def main(eq_thresh=1e-2):
    m = Model()
    m.train(DATA, 400)

    thresholds = np.linspace(0, 1e-1, 100)
    seqs = [DATA[0], DATA[1], "<s>cccccccc</s>"]
    seq_probs = [[] for _ in range(len(seqs))]

    # try to un-memorize seqs[0]
    grads = m.gradient(seqs[0])
    for t in tqdm(thresholds):
        with torch.no_grad():
            m.mask_params(grads, t)

        for i, seq in enumerate(seqs):
            seq_probs[i].append(m.prob(seq))

        m.restore_params()

    # trim the sequences to the point where they're all equal
    new_len = min(
        i
        for i in range(len(seq_probs[0]))
        if all(
            np.abs(np.log(seq_probs[j][i]) - np.log(seq_probs[0][i])) < eq_thresh
            for j in range(len(seqs))
        )
    )
    seq_probs = [seq[: round(new_len * 1.1)] for seq in seq_probs]
    thresholds = thresholds[: round(new_len * 1.1)]

    # plot the results
    for i, seq in enumerate(seqs):
        plt.plot(thresholds, np.log(seq_probs[i]), "-", label=seq)

    plt.xlabel("threshold")
    plt.ylabel("log probability")
    plt.title("mask params at different thresholds")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
