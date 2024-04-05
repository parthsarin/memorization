"""
File: sample_pile.py
--------------------

Samples paragraphs from the Pile, uniformly across different EM values and
saves them to a file.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
from collections import defaultdict
import numpy as np
import json
import feedparser
from newsplease import NewsPlease
import requests

EM_PREFIX_LEN = 50
EM_GENERATE_LEN = 50


def aggregate_article(article):
    paragraphs = article.split("\n")

    out = []
    curr = ""
    for sentence in paragraphs:
        proposal = curr + " " + sentence
        if len(proposal.split()) < 2 * (EM_PREFIX_LEN + EM_GENERATE_LEN):
            curr = proposal
        else:
            out.append(proposal)
            curr = sentence

    return out


def sample_from_article(article, model, tokenizer, aggregate=False):
    # break the sample into paragraphs
    if aggregate:
        paragraphs_raw = aggregate_article(article)
    else:
        paragraphs_raw = article.split("\n")
    paragraphs_raw = [x.strip() for x in paragraphs_raw if x.strip()]
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
    return (
        tokenizer.batch_decode(inputs, skip_special_tokens=True),
        correct_tokens,
        tokenizer.batch_decode(outputs, skip_special_tokens=True),
        tokenizer.batch_decode(expected_generation, skip_special_tokens=True),
    )


def main(args):
    # load the pile dataset and model
    pile = load_dataset("EleutherAI/the_pile_deduplicated")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    out = []

    # sample from the pile
    print("Sampling from the pile...")
    samples_per_em = defaultdict(int)
    for _ in range(args.max_pile_samples):
        sample = pile["train"][int(pile["train"].num_rows * np.random.rand())]
        paragraphs, num_correct, generations, expected = sample_from_article(
            sample["text"], model, tokenizer, aggregate=True
        )

        for p, em, g, ex in zip(paragraphs, num_correct, generations, expected):
            if samples_per_em[em] < args.samples_per_em:
                out.append(
                    {
                        "prefix": p,
                        "em": int(em),
                        "completion": g,
                        "expected": ex,
                        "origin": "pile",
                    }
                )
                samples_per_em[em] += 1

                with open(args.output_file, "w") as f:
                    json.dump(out, f, indent=2)

        if all([samples_per_em[em] >= args.samples_per_em for em in samples_per_em]):
            break

    # sample from non-pile data
    print("Sampling from non-pile data...")
    samples_per_em = defaultdict(int)
    feed = feedparser.parse("https://news.google.com/news/rss")
    entries = iter(feed.entries)
    for _ in range(args.max_non_pile_samples):
        url = next(entries).link
        r = requests.get(url)
        try:
            article = NewsPlease.from_url(r.url).get_dict()
            text = (
                article.get("text") if article.get("text") else article.get("maintext")
            )
        except Exception:
            text = None

        if not text:
            continue

        paragraphs, num_correct, generations, expected = sample_from_article(
            text, model, tokenizer, aggregate=True
        )

        for p, em, g, ex in zip(paragraphs, num_correct, generations, expected):
            if samples_per_em[em] < args.samples_per_em:
                out.append(
                    {
                        "prefix": p,
                        "em": int(em),
                        "completion": g,
                        "expected": ex,
                        "origin": r.url,
                    }
                )
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
