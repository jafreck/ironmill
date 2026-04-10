#!/usr/bin/env python3
"""MLX decode benchmark for comparison with ironmill.

Runs Qwen3.5-4B with INT4 quantization (group_size=128) to match
ironmill's configuration. Reports tok/s and latency.

Usage:
    python3 scripts/mlx_decode_bench.py [--quantize] [--tokens N]

Options:
    --quantize    Quantize the model to INT4 gs=128 first (only needed once)
    --tokens N    Number of decode tokens to generate (default: 50)
    --trace       Enable MLX metal tracing (saves to /tmp/mlx_trace.gputrace)
"""

import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="MLX decode benchmark")
    parser.add_argument("--quantize", action="store_true", help="Quantize model first")
    parser.add_argument("--tokens", type=int, default=50, help="Decode tokens")
    parser.add_argument("--trace", action="store_true", help="Enable Metal tracing")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Model name/path")
    parser.add_argument("--quantized-model", default="/tmp/Qwen3.5-4B-4bit-mlx",
                        help="Path for quantized model")
    args = parser.parse_args()

    import mlx.core as mx

    if args.quantize:
        print(f"Quantizing {args.model} to INT4 gs=128...")
        from mlx_lm import convert
        convert(
            args.model,
            quantize=True,
            q_bits=4,
            q_group_size=128,
            mlx_path=args.quantized_model,
        )
        print(f"Saved to {args.quantized_model}")

    # Load the quantized model
    model_path = args.quantized_model
    print(f"Loading {model_path}...")
    from mlx_lm import load
    model, tokenizer = load(model_path)

    # Warmup
    prompt = "The meaning of life is"
    print(f"Prompt: {prompt!r}")
    print(f"Generating {args.tokens} tokens...")

    # Warmup (3 tokens)
    inputs = mx.array(tokenizer.encode(prompt))
    logits = model(inputs[None])
    mx.eval(logits)

    if args.trace:
        # Enable Metal tracing via environment or MLX API
        print("Metal tracing: capturing to /tmp/mlx_trace.gputrace")
        mx.metal.start_capture("/tmp/mlx_trace.gputrace")

    # Timed decode loop
    input_ids = tokenizer.encode(prompt)

    # Prefill
    cache = None
    inputs = mx.array([input_ids])
    logits = model(inputs, cache=cache)
    mx.eval(logits)

    # Get KV cache from the model for decode
    # mlx_lm uses a generate stream approach; let's use that
    # Use stream_generate for proper KV cache handling with timing
    from mlx_lm import stream_generate

    tokens = []
    times = []

    gen = stream_generate(
        model, tokenizer, prompt,
        max_tokens=args.tokens,
    )

    for resp in gen:
        t1 = time.perf_counter()
        tokens.append(resp.token)
        if len(tokens) > 1:
            times.append(t1 - t0)
        t0 = t1

    if args.trace:
        mx.metal.stop_capture()
        print("Trace saved to /tmp/mlx_trace.gputrace")

    # Report
    # Skip first 3 tokens as warmup
    warmup = min(3, len(times))
    measured = times[warmup:]
    if not measured:
        print("Not enough tokens for measurement")
        return

    import statistics
    median_ms = statistics.median(measured) * 1000
    mean_ms = statistics.mean(measured) * 1000
    tok_s = 1000.0 / median_ms

    decoded_text = tokenizer.decode(tokens[:200])
    print(f"\nGenerated: {decoded_text[:200]}...")
    print(f"\n{'='*50}")
    print(f"MLX Decode Benchmark — {model_path}")
    print(f"{'='*50}")
    print(f"Tokens generated: {len(tokens)} ({warmup} warmup, {len(measured)} measured)")
    print(f"Median latency:   {median_ms:.2f} ms/tok")
    print(f"Mean latency:     {mean_ms:.2f} ms/tok")
    print(f"Throughput:       {tok_s:.1f} tok/s")

    # Per-token detail
    print(f"\nPer-token latencies (ms): {', '.join(f'{t*1000:.1f}' for t in times[:10])}...")

if __name__ == "__main__":
    main()
