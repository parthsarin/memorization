import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

EM_TARGET = 1
data = json.load(open("../out/ll_em_uniform_125m.json"))
data = [x for x in data if x["em"] == EM_TARGET]

df = []
for x in data:
    pl_3 = x["learning_log"][2]["logprob"]
    if pl_3 < -0.5:
        continue
    for s in x["learning_log"]:
        origin = "pile" if x["origin"] == "pile" else "non-pile"
        pl = s["prefix_len"]
        if pl >= 4:
            continue
        df.append({"pl": s["prefix_len"], "logprob": s["logprob"], "origin": origin})


df = pd.DataFrame(df)

sns.lineplot(data=df, x="pl", y="logprob", hue="origin")
plt.xlabel("Prefix length")
plt.ylabel("Log probability")
plt.xticks([1, 2, 3])
plt.title(f"Pile and non-pile samples with EM = {EM_TARGET}")

plt.savefig("../img/em_1.png", dpi=300)
