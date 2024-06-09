import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json

sns.set_theme(style="whitegrid")

qa_log = json.load(open("../out/qa_log_EleutherAI_gpt-neo-2.7B.json"))
# qa_log = json.load(open("../out/qa_log_mistralai_Mistral-7B-v0.1.json"))
print(f"Loaded {len(qa_log)} logs")


# metric to apply to extraction curve
def metric(log):
    # how should we normalize these based on the length of the question?
    best_logprob = max(d["logprob"] for d in log)
    worst_logprob = min(d["logprob"] for d in log)
    # thresh = worst_logprob + 0.6 * (best_logprob - worst_logprob)
    # reaches_target = [d["prefix_len"] for d in log if d["logprob"] > thresh]
    # return min(reaches_target) if reaches_target else float("inf")
    return best_logprob - worst_logprob


y = [d["correct_answer_logprob"] for d in qa_log]
x = [metric(d["extraction_log"]) for d in qa_log]

plt.figure(figsize=(12, 6))
# sns.scatterplot(x=x, y=y)
# errorbar
sns.regplot(x=x, y=y, x_estimator=np.mean, x_ci="ci")
plt.xlabel("Extraction metric")
plt.ylabel("Correct answer logprob")
plt.title("Extraction metric vs correct answer logprob")
plt.show()
