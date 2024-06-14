import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../out/extract_password_logprob.csv")
data = []

for i, row in df.iterrows():
    row = row.to_dict()
    for k, v in row.items():
        if k == "Unnamed: 0":
            continue
        data.append({"pl": k, "logprob": v})

data = pd.DataFrame(data)

df = pd.read_csv("../out/extract_incorrect_logprob.csv")
incorrect = []

for i, row in df.iterrows():
    row = row.to_dict()
    for k, v in row.items():
        if k == "Unnamed: 0":
            continue
        incorrect.append({"pl": k, "logprob": v})

incorrect = pd.DataFrame(incorrect)

sns.set_theme(style="whitegrid")

sns.lineplot(data=data, x="pl", y="logprob")
sns.lineplot(data=incorrect, x="pl", y="logprob")
plt.xlabel("Epochs")
plt.ylabel("Log probability")
plt.title("Password log probability")
plt.savefig("../img/password_logprob.png", dpi=300)
