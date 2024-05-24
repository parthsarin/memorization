import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

# get the data from the logs
data = json.load(open("out/sentence_embeddings_mistral.json"))
query_fit = np.array([x["query_logprob"] for x in data])
pos_sim = np.array([x["pos_sim"] for x in data])
neg_sim = np.array([x["neg_sim"] for x in data])

# regplot chart x = query_fit y = pos_sim, neg_sim
plt.figure(figsize=(10, 6))

sns.regplot(x=query_fit, y=pos_sim, color="blue", label="Positive")
sns.regplot(x=query_fit, y=neg_sim, color="red", label="Negative")

plt.xlabel("Query logprob")
plt.ylabel("Similarity")
plt.title("Query logprob vs similarity")
plt.legend()
plt.show()

# histplot
plt.figure(figsize=(10, 6))
sns.histplot(pos_sim, kde=True, color="blue", label="Positive")
sns.histplot(neg_sim, kde=True, color="red", label="Negative")

plt.xlabel("Similarity")
plt.ylabel("Density")
plt.title("Similarity between sentence embeddings")
plt.legend()
plt.show()
