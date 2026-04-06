#!/usr/bin/env python3
"""Compute perplexity of Qwen3.5-0.8B on WikiText-2 using HuggingFace transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import time
import argparse


def compute_perplexity(model_name="Qwen/Qwen3.5-0.8B", max_length=2048, stride=512, num_samples=None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"max_length={max_length}, stride={stride}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Load WikiText-2 test split
    print("\nLoading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")

    seq_len = encodings.input_ids.size(1)
    print(f"Total tokens: {seq_len}")

    nlls = []
    prev_end_loc = 0
    count = 0
    t0 = time.time()
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        count += 1

        if count % 50 == 0:
            elapsed = time.time() - t0
            curr_ppl = torch.exp(torch.stack(nlls).mean()).item()
            print(f"  window {count}: running PPL = {curr_ppl:.4f} ({elapsed:.1f}s elapsed)")

        if num_samples and count >= num_samples:
            break
        if end_loc == seq_len:
            break

    elapsed = time.time() - t0
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nPerplexity: {ppl.item():.4f}")
    print(f"Evaluated {count} windows in {elapsed:.1f}s")
    return ppl.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    ppl = compute_perplexity(
        model_name=args.model,
        max_length=args.max_length,
        stride=args.stride,
        num_samples=args.num_samples,
    )
    print(f"\n=== RESULT: {args.model} WikiText-2 PPL = {ppl:.4f} ===")
