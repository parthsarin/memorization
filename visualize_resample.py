import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# get underlying probability distribution
data = json.load(open("data/pile_and_non_pile_samples.json"))
ems = [x["em"] for x in data if x["origin"] == "pile"]
probs = collections.Counter(ems)
total = sum(probs.values())
probs = {k: v / total for k, v in probs.items()}

# resample pile log probs with the same distribution
data = json.load(open("out/ll_em_uniform_125m.json"))
pile_logprobs = collections.defaultdict(list)  # prefix length -> log probs

pile_data_raw = []
pile_sample_probs = []
for log in data:
    if log["origin"] != "pile":
        continue

    weight = probs[log["em"]]
    log = [(x["prefix_len"], x["logprob"]) for x in log["learning_log"]]

    pile_data_raw.append(log)
    pile_sample_probs.append(weight)

pile_sample_probs = np.array(pile_sample_probs) / np.sum(pile_sample_probs)

# resample
N = 10_000
pile_data = []
for _ in range(N):
    sample = np.random.choice(len(pile_data_raw), p=pile_sample_probs)

    for pl, lp in pile_data_raw[sample]:
        pile_logprobs[pl].append(lp)

# get non-pile log probs
non_pile_logprobs = collections.defaultdict(list)

for log in data:
    if log["origin"] == "pile":
        continue

    for x in log["learning_log"]:
        non_pile_logprobs[x["prefix_len"]].append(x["logprob"])


# plot
plt.figure(figsize=(10, 6))

for pl, logprobs in pile_logprobs.items():
    pile_logprobs[pl] = (np.mean(logprobs), np.std(logprobs))

for pl, logprobs in non_pile_logprobs.items():
    non_pile_logprobs[pl] = (np.mean(logprobs), np.std(logprobs))

k = np.array(list(pile_logprobs.keys()))
v = np.array(list(pile_logprobs.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="pile")

k = np.array(list(non_pile_logprobs.keys()))
v = np.array(list(non_pile_logprobs.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="non-pile")

plt.xlabel("Prefix length")
plt.ylabel("Log probability")
plt.legend()
plt.show()
