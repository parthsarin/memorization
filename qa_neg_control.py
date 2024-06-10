"""
File: qa_neg_control.py
-----------------------

Fine-tunes GPT-neo on some subset of the examples from the QA dataset and gets
the extraction curves on all of the examples.
"""
from learn_recall_prefix import *
import argparse
from datasets import load_dataset
import numpy as np
import torch
import torch.nn.functional as F
import wandb


def make_prompt(question, answer=True):
    prompt = f"Question: {question['question']}\n\n"
    choices = "ABCDEFGH"
    for l, c in zip(choices, question["choices"]):
        prompt += f"{l}. {c}\n"
    prompt = prompt.strip()
    if answer:
        prompt += f'\n\nAnswer: {choices[question["answer"]]}'
    return prompt


def main(args):
    pl = PrefixLearner(args.model)
    model_name = args.model
    data = load_dataset("hails/mmlu_no_train", "all")
    data = data["test"]
    N = min(args.num_questions, len(data))

    # get a random subset of the data
    data = [data[x] for x in np.random.permutation(len(data))[:N]]

    # fine-tune on the first 30% of the data
    bkpt = int(0.3 * N)
    ft_set = data[:bkpt]
    ft_set = [make_prompt(x) for x in ft_set]
    model = pl.model
    tokenizer = pl.tokenizer
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(args.epochs):
        total_loss = 0
        for i in range(0, len(ft_set), args.batch_size):
            batch = ft_set[i : i + args.batch_size]

            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(batch)

            wandb.log({"ft_batch_loss": loss.item()})

        total_loss /= len(ft_set)
        print(f"[epoch {epoch}] ft_loss: {total_loss}")
        wandb.log({"ft_loss": total_loss})

    print("\n\n====================\n\n")

    # get extraction curves on the data
    model.eval()
    qa_log = []
    for i, question in enumerate(data):
        d = {}

        did_fine_tune = i < bkpt
        d["did_fine_tune"] = did_fine_tune
        d["q"] = question

        print(f"Evaluating question {i} (fine-tuned: {did_fine_tune})")

        prompt = make_prompt(question, answer=False)
        print(f"[PROMPT]\n{prompt}\n[/PROMPT]")

        embeddings, extraction_log = pl.learn_prefix(prompt, max_recall_tokens=5)
        d["extraction_log"] = [l.to_dict() for l in extraction_log]

        choices = "ABCDEFGH"
        answer = choices[question["answer"]]
        print(f"* Answer: {answer}")

        prompt += "\n\nAnswer:"
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
        with open(f"qa_log_finetuned_{model_name.replace('/', '_')}.json", "w") as f:
            json.dump(qa_log, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tunes GPT-neo on some subset of the examples from the QA dataset and gets the extraction curves on all of the examples."
    )

    parser.add_argument("-m", "--model", type=str, default="EleutherAI/gpt-neo-125m")
    parser.add_argument("-n", "--num_questions", type=int, default=1000)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=40)

    args = parser.parse_args()
    main(args)
