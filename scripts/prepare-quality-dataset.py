"""Pre-tokenize WikiText-2 for perplexity evaluation.

Usage:
    pip install datasets transformers
    python scripts/prepare-quality-dataset.py [--seq-len 512] [--max-sequences 500]
"""
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize WikiText-2")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-sequences", type=int, default=500)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="tests/fixtures/quality/wikitext2-qwen3.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Loading WikiText-2 test split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    tokens = tokenizer.encode(full_text)
    print(f"Total tokens: {len(tokens)}")

    sequences = []
    for i in range(0, len(tokens) - args.seq_len, args.seq_len):
        sequences.append(tokens[i : i + args.seq_len])

    sequences = sequences[:args.max_sequences]

    output = {
        "name": f"WikiText-2 ({args.tokenizer} tokenizer)",
        "model": args.tokenizer.split("/")[-1],
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "seq_len": args.seq_len,
        "num_sequences": len(sequences),
        "eos_token_id": tokenizer.eos_token_id,
        "sequences": sequences,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)

    print(f"Wrote {len(sequences)} sequences ({args.seq_len} tokens each) to {args.output}")

if __name__ == "__main__":
    main()
