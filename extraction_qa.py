"""
Compare extraction curves with the model's performance on QA tasks.
"""
from datasets import load_dataset
from learn_recall_prefix import PrefixLearner
import json
import torch
import torch.nn.functional as F
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_prompt(question):
    prompt = f"Question: {question['question']}\n\n"
    choices = "ABCDEFGH"
    for l, c in zip(choices, question["choices"]):
        prompt += f"{l}. {c}\n"

    return prompt.strip(), choices[question["answer"]]


if __name__ == "__main__":
    pl = PrefixLearner("mistralai/Mistral-7B-v0.1")
    data = load_dataset("hails/mmlu_no_train", "all")
    data = data["test"]
    N = len(data)

    qa_log = []
    # go in a random order
    for i in np.random.permutation(N):
        question = data[int(i)]
        print(f"Evaluating question {i}")
        print("-" * 80)

        d = {}
        d["q"] = question

        # first, see how familiar the model is with the question
        prompt, answer = make_prompt(question)
        print(f"[PROMPT]\n{prompt}\n[/PROMPT]")

        embeddings, extraction_log = pl.learn_prefix(prompt, max_recall_tokens=3)
        d["extraction_log"] = [l.to_dict() for l in extraction_log]

        # then, see how well the model can answer the question
        print(f"* Answer: {answer}")

        prompt += "\n\nAnswer: "
        model = pl.model
        tokenizer = pl.tokenizer
        prompt = tokenizer(prompt, return_tensors="pt").to(device)
        completion = model(**prompt).logits.squeeze(0)[-1, :]
        logprobs = F.log_softmax(completion, dim=-1)

        answer = tokenizer(f"{answer}").input_ids[1]
        d["answer_logprobs"] = {
            a: logprobs[tokenizer(f"{a}").input_ids[1]].item() for a in "ABCDEFGH"
        }
        print(f"* Answer logprobs: {d['answer_logprob']}")

        d["correct_answer_logprob"] = logprobs[answer].item()
        print(f"* Correct answer logprob: {d['correct_answer_logprob']}")

        qa_log.append(d)
        wandb.log(d)
        with open("qa_log.json", "w") as f:
            json.dump(qa_log, f, indent=2)
