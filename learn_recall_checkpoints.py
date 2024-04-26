"""
File: learn_recall_checkpoints.py
---------------------------------

Learn recall prefixes across different model checkpoints to see when the model is memorizing a sequence.
"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from learn_recall_prefix import PrefixLearner
from sample_pile import sample_from_article
import numpy as np
import json

MODEL = "EleutherAI/pythia-160m"
CHECKPOINT_IDXS = [1, 4, 16, 64, 256, 512, 1000, 4000, 16000, 64000, 128000]
CHECKPOINTS = [f"step{idx}" for idx in CHECKPOINT_IDXS]
EM_THRESH = 40

PREFIX_LEARNERS = [PrefixLearner(MODEL, revision=step) for step in CHECKPOINTS]


def evaluate_checkpoints(prefix, target):
    out = {}
    for pl, step in zip(PREFIX_LEARNERS, CHECKPOINT_IDXS):
        print(f"{'='*10} Evaluating checkpoint {step} {'='*10}")
        emb, learning_log = pl.learn_prefix(target)
        out[step] = [l.to_dict() for l in learning_log]
    return out


def main(args):
    # load the pile dataset and model
    pile = load_dataset("EleutherAI/the_pile_deduplicated")
    N = pile["train"].num_rows
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # sample from the pile
    out = []
    mem_step_idx = 0
    while True:
        sample = pile["train"][int(N * np.random.rand())]
        try:
            paragraphs, num_correct, generations, expected = sample_from_article(
                sample["text"], model, tokenizer, aggregate=True
            )
        except Exception:
            continue

        for p, em, g, ex in zip(paragraphs, num_correct, generations, expected):
            if em >= EM_THRESH:
                print()
                print(f"Found a memorized sample: {repr(p)}")
                print(f"* EM: {em}")
                print(f"* Completion: {repr(g)}")
                print(f"* Expected: {repr(ex)}")

                # this sample was memorized by the final model -> let's see when it was memorized
                learning_by_step = evaluate_checkpoints(p, ex)
                out.append(
                    {
                        "prefix": p,
                        "em": int(em),
                        "completion": g,
                        "expected": ex,
                        "origin": "pile",
                        "learning_by_step": learning_by_step,
                    }
                )

                with open(args.output_file, "w") as f:
                    json.dump(out, f, indent=2)

            else:
                mem_step_idx += 1
                print(
                    f"No memorized samples found after {mem_step_idx} samples\r", end=""
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="recall_checkpoints.json")
    args = parser.parse_args()
    main(args)
