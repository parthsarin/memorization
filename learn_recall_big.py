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

import feedparser
from newsplease import NewsPlease
import requests
import random

MODEL = "EleutherAI/pythia-160m"
pl = PrefixLearner(MODEL)

feed = feedparser.parse("https://news.google.com/news/rss")
entries = iter(feed.entries)


def get_news_article():
    url = next(entries).link
    r = requests.get(url)
    try:
        article = NewsPlease.from_url(r.url).get_dict()
        text = article.get("text") if article.get("text") else article.get("maintext")
    except Exception:
        text = None

    return text, url


def main(args):
    # load the pile dataset and model
    pile = load_dataset("EleutherAI/the_pile_deduplicated")
    N = pile["train"].num_rows
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # sample from the pile
    out = []
    while True:
        if random.random() < 0.8:
            sample = pile["train"][int(N * np.random.rand())]
            origin = "pile"
        else:
            sample, origin = get_news_article()
        if not sample:
            continue
        try:
            paragraphs, num_correct, generations, expected = sample_from_article(
                sample["text"], model, tokenizer, aggregate=True
            )
        except Exception:
            continue

        for p, em, g, ex in zip(paragraphs, num_correct, generations, expected):
            print()
            print(f"Looking at sample: {repr(p)}")
            print(f"* EM: {em}")
            print(f"* Completion: {repr(g)}")
            print(f"* Expected: {repr(ex)}")

            emb, learning_log = pl.learn_prefix(ex)

            # this sample was memorized by the final model -> let's see when it was memorized
            out.append(
                {
                    "prefix": p,
                    "em": int(em),
                    "completion": g,
                    "expected": ex,
                    "origin": origin,
                    "learning_log": learning_log,
                }
            )

            with open(args.output_file, "w") as f:
                json.dump(out, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="recall_big.json")
    args = parser.parse_args()
    main(args)
