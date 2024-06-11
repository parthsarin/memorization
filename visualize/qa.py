import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd

sns.set_theme(style="whitegrid")

qa_log = json.load(open("../out/qa_log_finetuned_onesubj_EleutherAI_gpt-neo-125m.json"))
# qa_log = json.load(open("../out/qa_log_EleutherAI_gpt-neo-2.7B.json"))
# qa_log = json.load(open("../out/qa_log_mistralai_Mistral-7B-v0.1.json"))
print(f"Loaded {len(qa_log)} logs")
ft_subj = {d["q"]["subject"] for d in qa_log if d["did_fine_tune"]}
print(f"Fine-tuned on {ft_subj} subjects")


# metric to apply to extraction curve
def metric(log):
    # how should we normalize these based on the length of the question?
    return log[1]["logprob"] - log[0]["logprob"]
    best_logprob = max(d["logprob"] for d in log)
    worst_logprob = min(d["logprob"] for d in log)
    # thresh = worst_logprob + 0.6 * (best_logprob - worst_logprob)
    # reaches_target = [d["prefix_len"] for d in log if d["logprob"] > thresh]
    # return min(reaches_target) if reaches_target else float("inf")
    return best_logprob - worst_logprob


# df = []
# for d in qa_log:
#     x = {
#         "correct answer logprob": d["correct_answer_logprob"],
#         "finetuned": d["did_fine_tune"],
#         "color": "green" if d["did_fine_tune"] else "blue",
#     }
#     for l in d["extraction_log"]:
#         df.append({**x, "logprob": l["logprob"], "prefix_len": l["prefix_len"]})
# df = pd.DataFrame(df)

# plt.figure(figsize=(12, 6))
# # errorbar
# sns.regplot(
#     x="prefix_len",
#     y="logprob",
#     data=df[df.finetuned],
#     x_ci="ci",
#     color="green",
# )
# sns.regplot(
#     x="prefix_len",
#     y="logprob",
#     data=df[~df.finetuned],
#     x_ci="ci",
# )
# plt.xlabel("prefix len")
# plt.ylabel("logprob")
# plt.show()

# df = []
# for d in qa_log:
#     fine_tuned = d["did_fine_tune"]
#     extraction_metric = metric(d["extraction_log"])
#     correct_answer_logprob = d["correct_answer_logprob"]
#     df.append(
#         {
#             "extraction_metric": extraction_metric,
#             "correct_answer_logprob": correct_answer_logprob,
#             "finetuned": fine_tuned,
#             "color": "green" if fine_tuned else "blue",
#         }
#     )
# df = pd.DataFrame(df)

# plt.figure(figsize=(12, 6))
# # errorbar
# sns.regplot(
#     x="extraction_metric",
#     y="correct_answer_logprob",
#     data=df[df.finetuned],
#     x_ci="ci",
#     color="green",
# )

# sns.regplot(
#     x="extraction_metric",
#     y="correct_answer_logprob",
#     data=df[~df.finetuned],
#     x_ci="ci",
# )
# plt.xlabel("metric")
# plt.ylabel("correct answer logprob")
# plt.show()


df = []
for d in qa_log:
    ft = d["did_fine_tune"]
    x = {
        "correct answer logprob": d["correct_answer_logprob"],
    }
    cal = d["correct_answer_logprob"]

    # break this into buckets -1-0, -5 - -1, -10 - -5, and < -10, different colors
    if cal >= -1:
        x["bucket"] = "[-1, 0)"
        x["color"] = "green"
    elif cal >= -5:
        x["bucket"] = "[-5, -1)"
        x["color"] = "blue"
    elif cal >= -10:
        x["bucket"] = "[-10, -5)"
        x["color"] = "orange"
    else:
        x["bucket"] = "< -10"
        x["color"] = "red"

    for l in d["extraction_log"]:
        df.append({**x, "logprob": l["logprob"], "prefix_len": l["prefix_len"]})
df = pd.DataFrame(df)

plt.figure(figsize=(12, 6))
# errorbar
for bucket in df.bucket.unique():
    sns.regplot(
        x="prefix_len",
        y="logprob",
        data=df[df.bucket == bucket],
        x_ci="ci",
        color=df[df.bucket == bucket].color.values[0],
        label=bucket,
    )

plt.xlabel("prefix len")
plt.ylabel("logprob")
plt.legend()
plt.show()
