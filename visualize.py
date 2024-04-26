import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import collections

sns.set_theme(style="whitegrid")
plt.ion()

# get the data from the logs
learning_logs = json.load(open("out/learning_log_125m.json"))

# best loss for each EM
best_losses = collections.defaultdict(list)
best_pls = collections.defaultdict(list)
pile_losses = collections.defaultdict(list)
non_pile_losses = collections.defaultdict(list)
pile_ranges = []
non_pile_ranges = []
ems = [[], []]
for log in learning_logs:
    em = log["em"]
    losses = [l["loss"] for l in log["learning_log"]]
    pls = [l["prefix_len"] for l in log["learning_log"]]
    origin = log["origin"]

    range = np.max(losses) - np.min(losses)

    best_loss_idx = np.argmin(losses)

    best_losses[em].append(losses[best_loss_idx])
    best_pls[em].append(pls[best_loss_idx])
    if origin == "pile":
        add_to = pile_losses
        pile_ranges.append(range)
        # if em > 8:
        #     continue
    else:
        add_to = non_pile_losses
        non_pile_ranges.append(range)
    for pl, loss in zip(pls, losses):
        add_to[pl].append(loss)


print(f"Average pile EM: {np.mean(ems[0])}, std: {np.std(ems[0])}")
print(f"Average non-pile EM: {np.mean(ems[1])}, std: {np.std(ems[1])}")

for k, v in best_losses.items():
    best_losses[k] = (np.mean(v), np.std(v))

for k, v in best_pls.items():
    best_pls[k] = (np.mean(v), np.std(v))

# line plot the best loss for each EM
plt.figure(figsize=(10, 6))

k = np.array(list(best_losses.keys()))
v = np.array(list(best_losses.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="best loss")

plt.xlabel("EM")
plt.ylabel("loss")
plt.show()

# line plot the best prefix length for each EM
plt.figure(figsize=(10, 6))

k = np.array(list(best_pls.keys()))
v = np.array(list(best_pls.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="best prefix length")

plt.xlabel("EM")
plt.ylabel("prefix length")
plt.show()

# plot the loss in the pile and non-pile samples
plt.figure(figsize=(10, 6))

pile_losses = {k: (np.mean(v), np.std(v)) for k, v in pile_losses.items()}
non_pile_losses = {k: (np.mean(v), np.std(v)) for k, v in non_pile_losses.items()}

k = np.array(list(pile_losses.keys()))
v = np.array(list(pile_losses.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="pile")

k = np.array(list(non_pile_losses.keys()))
v = np.array(list(non_pile_losses.values()))
idx = np.argsort(k)

plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="non-pile")

plt.xlabel("prefix length")
plt.ylabel("loss")
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
