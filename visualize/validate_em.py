import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# get the data from the logs
logs = json.load(open("../out/ll_em_uniform_125m.json"))

df = []
for log in logs:
    em = log["em"]
    pl_1 = log["learning_log"][0]["logprob"]

    df.append({"em": em, "logprob": pl_1})

df = pd.DataFrame(df)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="em",
    y="logprob",
    marker="o",
)

plt.xlabel("EM")
plt.ylabel("Logprob")

plt.title("Logprob at prefix length 1")
plt.show()
