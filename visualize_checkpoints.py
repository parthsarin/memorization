import json
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

plt.ion()
data = json.load(open("recall_checkpoints.json"))

plt.figure(figsize=(10, 6))
data = data[0]["learning_by_step"]
for step, learning_log in data.items():
    pls = [l["prefix_len"] for l in learning_log]
    losses = [l["loss"] for l in learning_log]

    # thick lines
    # plt.plot(pls, losses, 'o-', label=f"step {step}")
    plt.plot(pls, losses, "o-", label=f"step {step}")

plt.xlabel("prefix length")
plt.ylabel("loss")
plt.legend()
plt.show()
breakpoint()
