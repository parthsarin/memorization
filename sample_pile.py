"""
File: sample_pile.py
--------------------

Samples paragraphs from the Pile, uniformly across different EM values and
saves them to a file.
"""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from collections import defaultdict
import numpy as np
import json

EM_PREFIX_LEN = 50
EM_GENERATE_LEN = 50


def main(args):
    # load the pile dataset and model
    pile = load_dataset("EleutherAI/the_pile_deduplicated")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # keep track of how many samples we have for each EM value
    samples_per_em = defaultdict(int)
    out = []

    # sample from the pile
    for _ in range(args.max_pile_samples):
        sample = pile["train"][int(pile["train"].num_rows * np.random.rand())]

        # break the sample into paragraphs
        paragraphs_raw = sample["text"].split("\n")
        paragraphs = tokenizer(
            paragraphs_raw,
            return_tensors="pt",
            max_length=EM_PREFIX_LEN + EM_GENERATE_LEN,
            truncation=True,
            padding="max_length",
        )
        paragraphs_raw = tokenizer.batch_decode(
            paragraphs["input_ids"], skip_special_tokens=True
        )

        # generate 50 tokens from the first 50 tokens of the paragraph
        inputs = paragraphs["input_ids"][:, :EM_PREFIX_LEN]
        expected_generation = paragraphs["input_ids"][:, EM_PREFIX_LEN:]
        outputs = model.generate(
            inputs, max_length=100, num_return_sequences=1, do_sample=False
        )

        # count how many of the generated tokens are the same as the original tokens
        outputs = outputs[:, EM_PREFIX_LEN:]
        correct_tokens = (outputs == expected_generation).sum(dim=1).numpy()

        for p, em in zip(paragraphs_raw, correct_tokens):
            if samples_per_em[em] < args.samples_per_em:
                out.append({"text": p, "em": em})
                samples_per_em[em] += 1

                with open(args.output_file, "w") as f:
                    json.dump(out, f, indent=2)

        if all([samples_per_em[em] >= args.samples_per_em for em in samples_per_em]):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--max-pile-samples", type=int, default=10_000)
    parser.add_argument("--max-non-pile-samples", type=int, default=100)
    parser.add_argument("--samples-per-em", type=int, default=10)
    parser.add_argument("--output-file", type=str, default="pile_samples.json")

    args = parser.parse_args()
    main(args)
