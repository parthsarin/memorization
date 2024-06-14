import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


sns.set_theme(style="whitegrid")

# get underlying probability distribution
# data = json.load(open("../data/pile_and_non_pile_samples.json"))
# ems = [x["em"] for x in data if x["origin"] == "pile"]
# probs = collections.Counter(ems)
# total = sum(probs.values())
# probs = {k: v / total for k, v in probs.items()}

# resample pile log probs with the same distribution
data = json.load(open("../out/ll_em_uniform_125m.json"))
pile_logprobs = collections.defaultdict(list)  # prefix length -> log probs
pile_metric = []  # custom metric to determine signal

for log in json.load(open("../out/ll_em_nonuniform_125m.json")):
    if log["origin"] != "pile":
        continue
    ll = log["learning_log"]
    metric = ll[1]["logprob"] - ll[0]["logprob"]
    if metric < 0:
        continue
    pile_metric.append(metric)

# pile_data_raw = []
# pile_sample_probs = []
# for log in data:
#     if log["origin"] != "pile":
#         continue

#     weight = probs[log["em"]]
#     log = [(x["prefix_len"], x["logprob"], log["em"]) for x in log["learning_log"]]

#     pile_data_raw.append(log)
#     pile_sample_probs.append(weight)

# pile_sample_probs = np.array(pile_sample_probs) / np.sum(pile_sample_probs)

# # resample
# N = 10_000
# pile_data = []
# for _ in range(N):
#     sample = np.random.choice(len(pile_data_raw), p=pile_sample_probs)

#     x = pile_data_raw[sample]
#     metric = x[1][1] - x[0][1]
#     # metric = -x[0][1]
#     if metric < 0:
#         continue
#     pile_metric.append(metric)

#     for pl, lp, em in pile_data_raw[sample]:
#         pile_logprobs[pl].append(lp)

# get non-pile log probs
non_pile_logprobs = collections.defaultdict(list)
non_pile_metric = []

for log in data:
    if log["origin"] == "pile":
        continue

    x = log["learning_log"]
    non_pile_metric.append(x[1]["logprob"] - x[0]["logprob"])

    for x in log["learning_log"]:
        non_pile_logprobs[x["prefix_len"]].append(x["logprob"])


# bar graph of the metric
plt.figure(figsize=(6, 6))

df = []
for m in pile_metric:
    df.append(
        {
            "origin": "pile",
            "metric": m,
        }
    )

for m in non_pile_metric:
    df.append(
        {
            "origin": "non-pile",
            "metric": m,
        }
    )

df = pd.DataFrame(df)

print(f"avg metric on pile: {np.mean(df[df['origin'] == 'pile']['metric']):.2f}")
print(
    f"avg metric on non-pile: {np.mean(df[df['origin'] == 'non-pile']['metric']):.2f}"
)

p_val = stats.ttest_ind(
    df[df["origin"] == "pile"]["metric"], df[df["origin"] == "non-pile"]["metric"]
)[1]
print(f"t-test p-value: {p_val}")

p_val = stats.ks_2samp(
    df[df["origin"] == "pile"]["metric"], df[df["origin"] == "non-pile"]["metric"]
)
print(f"ks-test p-value: {p_val}")

# pile_metric = ems
# non_pile_metric = [x["em"] for x in data if x["origin"] != "pile"]
sns.barplot(df, x="origin", y="metric", errorbar="ci")
# sns.boxplot(data=df, x="origin", y="metric")
# sns.displot(
#     df,
#     x="metric",
#     hue="origin",
#     kind="kde",
#     fill=True,
#     common_norm=False,
#     common_grid=True,
# )

plt.xlabel("Sample origin")
# plt.ylabel("Average EM score")
plt.ylabel("Slope of extraction curve at prefix length 1")

plt.savefig("../img/pile_detector.png", dpi=300)

# plot
# plt.figure(figsize=(10, 6))

# for pl, logprobs in pile_logprobs.items():
#     pile_logprobs[pl] = (np.mean(logprobs), np.std(logprobs))

# for pl, logprobs in non_pile_logprobs.items():
#     non_pile_logprobs[pl] = (np.mean(logprobs), np.std(logprobs))

# k = np.array(list(pile_logprobs.keys()))
# v = np.array(list(pile_logprobs.values()))
# idx = np.argsort(k)

# plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="pile")

# k = np.array(list(non_pile_logprobs.keys()))
# v = np.array(list(non_pile_logprobs.values()))
# idx = np.argsort(k)

# plt.errorbar(k[idx], v[idx, 0], yerr=v[idx, 1], marker="o", label="non-pile")

# plt.xlabel("Prefix length")
# plt.ylabel("Log probability")
# plt.legend()
# plt.show()
