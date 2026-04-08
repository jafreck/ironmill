# Benchmark Reproduction Guide

## Hardware

- Apple M2 Max, 64 GB unified memory
- macOS 15.x, Metal GPU Family: Apple 8

## Model

- Qwen3.5-4B (`Qwen/Qwen3.5-4B` from HuggingFace)
- 32 layers (8 standard attention, 24 GDN recurrent)
- hidden_size=2560, head_dim=256, vocab=248320

## Prerequisites

```bash
# Download model weights
hf download Qwen/Qwen3.5-4B --cache-dir ~/.cache/huggingface/hub
ln -s ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/<hash> models/Qwen3.5-4B

# Build
cargo build --release -p ironmill-bench --features metal
```

## 1. GPU Memory + PPL (AWQ-INT4 + TQ-INT8)

### Step 1: AWQ Calibration

Captures per-projection activation magnitudes from 50 WikiText-2
sequences. Produces `awq_magnitudes.json` with per-channel importance
scores for all 200 projections (Q/K/V/gate/up at hidden_size=2560,
O_proj at 4096, down_proj at 9216).

```bash
cargo run --release --example awq_calibrate --features metal -p ironmill-bench -- \
    models/Qwen3.5-4B \
    tests/fixtures/quality/wikitext2-qwen35.json \
    /tmp/awq_calib
```

### Step 2: Benchmark

Config file (`/tmp/bench_awq.toml`):

```toml
[[model]]
name = "Qwen3.5-4B"
path = "unused"
model_dir = "models/Qwen3.5-4B"

[[optimization]]
name = "int4-naive"
int4 = true
kv_quant = "turbo-int8"
max_seq_len = 2048

[[optimization]]
name = "int4-awq"
int4 = true
awq_calib_dir = "/tmp/awq_calib"
kv_quant = "turbo-int8"
max_seq_len = 2048

[settings]
iterations = 5
warmup = 2
runs = 1
backends = ["metal"]
```

```bash
cargo run --release -p ironmill-bench --features metal -- \
    --config /tmp/bench_awq.toml -b metal \
    -i 5 -w 2 -r 1 \
    --perplexity --perplexity-sequences 1 \
    --perplexity-dataset tests/fixtures/quality/wikitext2-qwen35.json
```

### Results

| Config | PPL | ΔPPL vs FP16 | GPU MB | ΔMem vs FP16 |
|--------|-----|-------------|--------|-------------|
| FP16 baseline | 8.50 | — | 9,543 | — |
| INT4 naive + TQ-INT8 | 9.39 | +10.5% | 2,898 | -70% |
| INT4 AWQ + TQ-INT8 | 9.22 | +8.5% | 2,900 | -70% |

PPL measured on 1 WikiText-2 sequence (511 tokens). For more reliable
numbers, use `--perplexity-sequences 50`.

## 2. Decode Speed at Long Context

Uses FlashDecoding split-KV attention to parallelize decode across
GPU cores. Config file (`configs/qwen35-4b-kv-context-scaling.toml`):

```bash
cargo run --release -p ironmill-bench --features metal -- \
    --config configs/qwen35-4b-kv-context-scaling.toml -b metal \
    -i 20 -w 5 -r 1 \
    --context-lengths 128,1024,4096,16384
```

The `--context-lengths` flag prefills N tokens (untimed), then
measures per-token decode latency at that KV cache depth.

### Results

| Context | FP16 KV | TQ-INT4 KV |
|---------|---------|-----------|
| 128 | 58ms (17.2 tok/s) | 60ms (16.7 tok/s) |
| 1,024 | 59ms (16.9 tok/s) | 63ms (15.9 tok/s) |
| 4,096 | 59ms (16.8 tok/s) | 63ms (16.0 tok/s) |
| 16,384 | 62ms (16.2 tok/s) | 72ms (13.9 tok/s) |

FP16 decode is flat up to 16K thanks to FlashDecoding. TQ-INT4 shows
a mild slope from quantized attention's per-position dequant cost.

### Before FlashDecoding (for comparison)

| Context | FP16 KV (old) | FP16 KV (new) | Speedup |
|---------|--------------|---------------|---------|
| 1,024 | 142ms | 59ms | 2.4× |
| 4,096 | 395ms | 59ms | 6.7× |
| 16,384 | 1,412ms | 62ms | 22.9× |

## 3. Prefill Speed

```bash
cargo run --release -p ironmill-bench --features metal -- \
    --config configs/qwen35-4b-kv-context-scaling.toml -b metal \
    -i 20 -w 5 -r 1 \
    --prefill-bench
```

The `--prefill-bench` flag measures prefill throughput at input
lengths 128, 512, 1024, 2048, 4096.

### Results

| Prefill length | FP16 | TQ-INT8 | TQ-INT4 |
|---------------|------|---------|---------|
| 128 | 243 tok/s | 235 tok/s | 237 tok/s |
| 512 | 289 tok/s | 274 tok/s | 280 tok/s |
| 1,024 | 289 tok/s | 267 tok/s | 278 tok/s |
| 2,048 | 278 tok/s | 250 tok/s | 263 tok/s |
| 4,096 | 247 tok/s | 214 tok/s | 230 tok/s |

FP16 is fastest for prefill (~290 tok/s peak). TQ adds 5-10% overhead
from quantized KV cache writes.

## 4. Full Suite (all optimizations + PPL)

The comprehensive config at `configs/qwen35-4b-optimization-matrix.toml` benchmarks
all five configurations (FP16, FP16+TQ-INT8, FP16+TQ-INT4, D2Q3+TQ-INT8,
INT4+TQ-INT8):

```bash
cargo run --release -p ironmill-bench --features metal -- \
    --config configs/qwen35-4b-optimization-matrix.toml -b metal \
    -i 20 -w 5 -r 3 \
    --perplexity --perplexity-sequences 1 \
    --perplexity-dataset tests/fixtures/quality/wikitext2-qwen35.json
```

### Results

| Config | PPL | tok/s | ms/tok | GPU MB | ΔPPL | ΔMem |
|--------|-----|-------|--------|--------|------|------|
| fp16-baseline | 8.50 | 17.7 | 56.4 | 9,543 | — | — |
| fp16-tq-int8 | 8.51 | 17.7 | 56.6 | 9,432 | +0.1% | -1% |
| fp16-tq-int4 | 8.57 | 17.8 | 56.3 | 9,368 | +0.8% | -2% |
| d2q3-tq-int8 | 12.56 | 16.3 | 61.5 | 6,226 | +47.7% | -35% |
| int4-tq-int8 | 9.34 | 20.5 | 48.9 | 4,676 | +9.8% | -51% |

Note: The INT4 results in this table use the older GPU memory layout
(before the INT4 embedding and lm_head dedup fixes). With those fixes,
INT4+TQ-INT8 uses 2,898 MB (-70%).

## Benchmark CLI Reference

| Flag | Description |
|------|-------------|
| `--config <file>` | TOML config with model + optimization matrix |
| `-b metal` | Use Metal GPU backend |
| `-i N` | Decode iterations per measurement |
| `-w N` | Warmup iterations (prefilled as batch) |
| `-r N` | Number of full runs for statistics |
| `--prefill-bench` | Measure prefill throughput at 128-4096 tokens |
| `--context-lengths N,N,...` | Decode-at-context: prefill N tokens, then measure decode |
| `--perplexity` | Run PPL evaluation after latency benchmarks |
| `--perplexity-sequences N` | Number of WikiText-2 sequences for PPL (default: 50) |
| `--perplexity-dataset <path>` | Pre-tokenized dataset path |
