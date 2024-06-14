import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

data = json.load(open("../out/leak_train_data.json"))

df = []
ems = [d["ems"] for d in data]
for em in ems:
    baseline = em["0"]
    diffs = {pl: em - baseline for pl, em in em.items()}
    df.append(diffs)
df = pd.DataFrame(df)


plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Difference in Exact Matches from Prompt 0")
plt.ylabel("Difference in Exact Matches")
plt.xlabel("Prompt")
plt.show()
