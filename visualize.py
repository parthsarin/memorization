import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import collections

sns.set_theme(style="whitegrid")
plt.ion()

# get the data from the logs
# learning_logs = json.load(open("out/learning_log_2_7b.json"))
learning_logs = json.load(open("out/learning_log_125m.json"))

# best loss for each EM
best_logprobs = collections.defaultdict(list)
best_pls = collections.defaultdict(list)
pile_logprobs = collections.defaultdict(list)
non_pile_logprobs = collections.defaultdict(list)
pl_1_logprobs = collections.defaultdict(list)
pile_ranges = []
non_pile_ranges = []
ems = [[], []]
for log in learning_logs:
    em = log["em"]
    logprobs = [l["logprob"] for l in log["learning_log"]]
    pls = [l["prefix_len"] for l in log["learning_log"]]
    origin = log["origin"]

    range = np.max(logprobs) - np.min(logprobs)

    best_logprob_idx = np.argmax(logprobs)

    best_logprobs[em].append(logprobs[best_logprob_idx])
    best_pls[em].append(pls[best_logprob_idx])
    pl_1_logprobs[em].append(logprobs[0])
    if origin == "pile":
        add_to = pile_logprobs
        pile_ranges.append(range)
    else:
        add_to = non_pile_logprobs
        non_pile_ranges.append(range)
    for pl, logprob in zip(pls, logprobs):
        add_to[pl].append(logprob)


print(f"Average pile EM: {np.mean(ems[0])}, std: {np.std(ems[0])}")
print(f"Average non-pile EM: {np.mean(ems[1])}, std: {np.std(ems[1])}")

for k, v in best_logprobs.items():
    best_logprobs[k] = (np.mean(v), np.std(v))

for k, v in pl_1_logprobs.items():
    pl_1_logprobs[k] = (np.mean(v), np.std(v))

for k, v in best_pls.items():
    best_pls[k] = (np.mean(v), np.std(v))

# line plot the best logprob for each EM
plt.figure(figsize=(10, 6))

k = np.array(list(pl_1_logprobs.keys()))
v = np.array(list(pl_1_logprobs.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="best logprob")

plt.xlabel("EM")
plt.ylabel("logprob")
plt.savefig("em_pl_loss.png", dpi=300)
plt.show()

# line plot the best prefix length for each EM
# plt.figure(figsize=(10, 6))

# k = np.array(list(best_pls.keys()))
# v = np.array(list(best_pls.values()))
# idx = np.argsort(k)

# plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="best prefix length")

# plt.xlabel("EM")
# plt.ylabel("prefix length")
# plt.show()

# plot the logprob in the pile and non-pile samples
plt.figure(figsize=(10, 6))

pile_logprobs = {k: (np.mean(v), np.std(v)) for k, v in pile_logprobs.items()}
non_pile_logprobs = {k: (np.mean(v), np.std(v)) for k, v in non_pile_logprobs.items()}

k = np.array(list(pile_logprobs.keys()))
v = np.array(list(pile_logprobs.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="pile")

k = np.array(list(non_pile_logprobs.keys()))
v = np.array(list(non_pile_logprobs.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="non-pile")

plt.xlabel("prefix length")
plt.ylabel("logprob")
plt.legend()
plt.show()

# compare the ranges
plt.figure(figsize=(10, 6))

pile_ranges = np.array(pile_ranges)
non_pile_ranges = np.array(non_pile_ranges)

m, s = np.mean(pile_ranges), np.std(pile_ranges)
plt.errorbar(0, m, yerr=s, marker="o", label="pile")

m, s = np.mean(non_pile_ranges), np.std(non_pile_ranges)
plt.errorbar(1, m, yerr=s, marker="o", label="non-pile")

plt.xlabel("range")
plt.ylabel("count")
plt.legend()
plt.show()

breakpoint()
