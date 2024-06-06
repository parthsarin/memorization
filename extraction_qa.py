"""
Compare extraction curves with the model's performance on QA tasks.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from learn_recall_prefix import PrefixLearner
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_prompt(question):
    prompt = f"Question: {question['question']}\n\n"
    choices = "ABCDEFGH"
    for l, c in zip(choices, question["choices"]):
        prompt += f"{l}. {c}\n"
    return prompt.strip(), choices[question["answer"]]


def main(model_name, num_questions):
    pl = PrefixLearner(model_name)
    data = load_dataset("hails/mmlu_no_train", "all")
    data = data["test"]
    N = min(num_questions, len(data))

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

        embeddings, extraction_log = pl.learn_prefix(prompt, max_recall_tokens=5)
        d["extraction_log"] = [l.to_dict() for l in extraction_log]

        # then, see how well the model can answer the question
        print(f"* Answer: {answer}")

        prompt += "\n\nAnswer:"
        model = pl.model
        tokenizer = pl.tokenizer
        prompt = tokenizer(prompt, return_tensors="pt").to(device)
        completion = model(**prompt).logits.squeeze(0)[-1, :]
        logprobs = F.log_softmax(completion, dim=-1)

        answer = tokenizer(f" {answer}").input_ids[-1]
        d["answer_logprobs"] = {
            a: logprobs[tokenizer(f" {a}").input_ids[-1]].item() for a in "ABCDEFGH"
        }
        print(f"* Answer logprobs: {d['answer_logprobs']}")

        d["correct_answer_logprob"] = logprobs[answer].item()
        print(f"* Correct answer logprob: {d['correct_answer_logprob']}")

        qa_log.append(d)
        wandb.log(d)
        with open(f"qa_log_{model_name.replace('/', '_')}.json", "w") as f:
            json.dump(qa_log, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare extraction curves with the model's performance on QA tasks."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The model to use.",
        default="EleutherAI/gpt-neo-2.7B",
    )
    parser.add_argument(
        "-n",
        "--num_questions",
        type=int,
        default=100,
        help="Number of questions to evaluate.",
    )
    args = parser.parse_args()

    main(args.model, args.num_questions)
