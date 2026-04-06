#!/usr/bin/env python3
"""Compare HF Qwen3.5-0.8B layer intermediates with Metal GPU readback.

Hooks into the actual HF model to capture hidden states at key points:
  - After each layer's input norm
  - After each GDN/attention output (before residual)
  - After residual + post-attention norm (input to FFN)
  - After FFN output (before residual)
  - After final layer + final norm (lm_head input)

Also computes per-position cross-entropy to characterize the scale drift.

Usage:
    cd /Users/jacobfreck/Source/ironmill-qwen35
    python3 scripts/gdn_layer_compare.py [--tokens 20] [--save-dir /tmp/hf_ref]
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=20,
                        help="Number of tokens to process")
    parser.add_argument("--save-dir", type=str, default="/tmp/hf_ref",
                        help="Directory to save reference values")
    parser.add_argument("--model-dir", type=str, default="models/Qwen3.5-0.8B",
                        help="Model directory")
    parser.add_argument("--compare-dir", type=str, default=None,
                        help="Directory with Metal readback files to compare")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    print(f"Model: {model_dir}")

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,  # FP32 reference
        trust_remote_code=True,
    )
    model.eval()

    # ── Get tokens from wikitext2 fixture ──
    fixture = Path("tests/fixtures/quality/wikitext2-qwen35.json")
    with open(fixture) as f:
        data = json.load(f)
    token_ids = data["sequences"][0][:args.tokens]
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    T = len(token_ids)
    print(f"Tokens: {T} (first 5: {token_ids[:5]})")

    # ── Register hooks to capture intermediates ──
    captured = {}

    def make_pre_hook(layer_idx):
        """Capture input to each decoder layer (= after previous layer's norm)."""
        def hook(module, args):
            x = args[0]  # hidden_states
            captured[f"layer{layer_idx}_input"] = x.detach().clone()
        return hook

    def make_post_hook(layer_idx):
        """Capture output of each decoder layer (= after residual add)."""
        def hook(module, args, output):
            x = output[0]  # hidden_states
            captured[f"layer{layer_idx}_output"] = x.detach().clone()
        return hook

    # Get the model layers (handle VLM nesting)
    if hasattr(model, 'model'):
        base = model.model
        if hasattr(base, 'language_model'):
            base = base.language_model.model
    else:
        base = model.model

    layers = base.layers

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_pre_hook(i), with_kwargs=False)
        layer.register_forward_hook(make_post_hook(i))

    # Also hook the GDN internals for layer 0 (which is a GDN layer)
    gdn_captured = {}

    layer0 = layers[0]
    if hasattr(layer0, 'linear_attn'):
        gdn = layer0.linear_attn

        # Hook the GDN sub-modules
        def capture_after_conv(module, args, output):
            gdn_captured["conv_out"] = output.detach().clone()

        def capture_in_proj_qkv(module, args, output):
            gdn_captured["qkv_proj"] = output.detach().clone()

        def capture_in_proj_z(module, args, output):
            gdn_captured["z_proj"] = output.detach().clone()

        def capture_in_proj_a(module, args, output):
            gdn_captured["a_proj"] = output.detach().clone()

        def capture_in_proj_b(module, args, output):
            gdn_captured["b_proj"] = output.detach().clone()

        def capture_norm(module, args, output):
            gdn_captured["gated_output"] = output.detach().clone()

        def capture_out_proj(module, args, output):
            gdn_captured["out_proj"] = output.detach().clone()

        gdn.in_proj_qkv.register_forward_hook(capture_in_proj_qkv)
        gdn.in_proj_z.register_forward_hook(capture_in_proj_z)
        gdn.in_proj_a.register_forward_hook(capture_in_proj_a)
        gdn.in_proj_b.register_forward_hook(capture_in_proj_b)
        gdn.norm.register_forward_hook(capture_norm)
        gdn.out_proj.register_forward_hook(capture_out_proj)

        # Hook the recurrent function to capture post-recurrent output
        orig_chunk_fn = gdn.chunk_gated_delta_rule
        orig_recurrent_fn = gdn.recurrent_gated_delta_rule

        def hooked_chunk(*a, **kw):
            result = orig_chunk_fn(*a, **kw)
            gdn_captured["recurrent_out"] = result[0].detach().clone()
            gdn_captured["recurrent_state"] = (
                result[1].detach().clone() if result[1] is not None else None
            )
            return result

        def hooked_recurrent(*a, **kw):
            result = orig_recurrent_fn(*a, **kw)
            gdn_captured["recurrent_out"] = result[0].detach().clone()
            gdn_captured["recurrent_state"] = (
                result[1].detach().clone() if result[1] is not None else None
            )
            return result

        gdn.chunk_gated_delta_rule = hooked_chunk
        gdn.recurrent_gated_delta_rule = hooked_recurrent

    # ── Also run the SEQUENTIAL recurrent for comparison ──
    # This tells us if chunk vs recurrent gives different results.
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_recurrent_gated_delta_rule,
    )

    # ── Forward pass ──
    print("\nRunning HF forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0]  # [T, vocab_size]

    print(f"Logits shape: {logits.shape}")

    # ── Per-position cross-entropy ──
    print("\n=== Per-position Cross-Entropy ===")
    for t in range(min(T - 1, 40)):
        target = token_ids[t + 1]
        log_probs = F.log_softmax(logits[t].float(), dim=-1)
        ce = -log_probs[target].item()
        argmax = logits[t].argmax().item()
        top_val = logits[t].max().item()
        print(f"  pos={t:3d}: argmax={argmax:6d} max={top_val:8.3f} "
              f"CE={ce:7.4f} target={target}")

    total_ce = 0.0
    count = 0
    for t in range(T - 1):
        target = token_ids[t + 1]
        log_probs = F.log_softmax(logits[t].float(), dim=-1)
        ce = -log_probs[target].item()
        total_ce += ce
        count += 1
    ppl = math.exp(total_ce / count) if count > 0 else float('inf')
    print(f"\nHF PPL ({count} positions): {ppl:.4f}")

    # ── Save layer intermediates ──
    print(f"\nSaving intermediates to {save_dir}/")

    # Save final logits
    np.save(save_dir / "logits.npy", logits.float().numpy())

    # Save per-layer hidden states
    for key, val in sorted(captured.items()):
        arr = val[0].float().numpy()  # Remove batch dim
        np.save(save_dir / f"{key}.npy", arr)
        first5 = arr[0, :5].tolist() if arr.ndim >= 2 else arr[:5].tolist()
        print(f"  {key}: shape={arr.shape} "
              f"mean={arr.mean():.6f} std={arr.std():.6f} "
              f"first5={first5}")

    # Save GDN layer 0 internals
    print("\n=== GDN Layer 0 Internals ===")
    for key, val in sorted(gdn_captured.items()):
        if val is None:
            print(f"  {key}: None")
            continue
        arr = val.float().numpy()
        if arr.ndim > 2:
            arr = arr[0]  # Remove batch dim
        np.save(save_dir / f"gdn0_{key}.npy", arr)
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr
        print(f"  {key}: shape={arr.shape} "
              f"mean={flat.mean():.6f} std={flat.std():.6f}")

    # ── Also run sequential recurrent for chunk vs recurrent comparison ──
    print("\n=== Chunk vs Recurrent comparison (layer 0) ===")
    if hasattr(layer0, 'linear_attn'):
        gdn_mod = layer0.linear_attn
        # Reconstruct inputs to GDN layer 0
        # The layer input is the input_norm applied to the hidden state
        layer0_input = captured["layer0_input"][0]  # [T, hidden]
        normed = layer0.input_layernorm(layer0_input.unsqueeze(0))

        # Run projections
        qkv = gdn_mod.in_proj_qkv(normed)
        z = gdn_mod.in_proj_z(normed)
        b = gdn_mod.in_proj_b(normed)
        a = gdn_mod.in_proj_a(normed)

        # Conv1d
        qkv_t = qkv.transpose(1, 2)
        if gdn_mod.causal_conv1d_fn is not None:
            conv_out = gdn_mod.causal_conv1d_fn(
                x=qkv_t,
                weight=gdn_mod.conv1d.weight.squeeze(1),
                bias=gdn_mod.conv1d.bias,
                activation=gdn_mod.activation,
            )
        else:
            conv_out = F.silu(gdn_mod.conv1d(qkv_t)[:, :, :T])
        conv_out = conv_out.transpose(1, 2)

        # Split
        q, k, v = torch.split(conv_out, [
            gdn_mod.key_dim, gdn_mod.key_dim, gdn_mod.value_dim
        ], dim=-1)
        q = q.reshape(1, T, -1, gdn_mod.head_k_dim)
        k = k.reshape(1, T, -1, gdn_mod.head_k_dim)
        v = v.reshape(1, T, -1, gdn_mod.head_v_dim)

        beta = b.sigmoid()
        g = -gdn_mod.A_log.float().exp() * F.softplus(a.float() + gdn_mod.dt_bias)

        # Run sequential recurrent version
        with torch.no_grad():
            seq_out, seq_state = torch_recurrent_gated_delta_rule(
                q.float(), k.float(), v.float(), g, beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        # Compare with chunk version (captured earlier)
        chunk_out = gdn_captured.get("recurrent_out")
        if chunk_out is not None:
            chunk_np = chunk_out[0].float().numpy()
            seq_np = seq_out[0].float().numpy()
            diff = np.abs(chunk_np - seq_np)
            print(f"  Chunk output shape: {chunk_np.shape}")
            print(f"  Recurrent output shape: {seq_np.shape}")
            print(f"  Max abs diff: {diff.max():.8f}")
            print(f"  Mean abs diff: {diff.mean():.8f}")
            if diff.max() > 0.001:
                print("  ⚠️  CHUNK vs RECURRENT MISMATCH!")
                for t in range(min(T, 5)):
                    print(f"    pos {t}: chunk[:5]={chunk_np[t,:5].tolist()}")
                    print(f"    pos {t}: seq[:5]  ={seq_np[t,:5].tolist()}")
            else:
                print("  ✓ Chunk and recurrent match")

        # Save sequential recurrent output
        np.save(save_dir / "gdn0_seq_recurrent_out.npy", seq_out[0].float().numpy())
        if seq_state is not None:
            np.save(save_dir / "gdn0_seq_recurrent_state.npy", seq_state.float().numpy())

    # ── Compare with Metal readback if available ──
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        print(f"\n=== Comparing with Metal readback from {compare_dir} ===")
        for npy_file in sorted(compare_dir.glob("*.npy")):
            key = npy_file.stem
            hf_file = save_dir / f"{key}.npy"
            if hf_file.exists():
                hf_val = np.load(hf_file)
                metal_val = np.load(npy_file)
                if hf_val.shape == metal_val.shape:
                    diff = np.abs(hf_val - metal_val)
                    corr = np.corrcoef(hf_val.flatten(), metal_val.flatten())[0, 1]
                    scale = np.std(metal_val) / np.std(hf_val) if np.std(hf_val) > 0 else float('inf')
                    print(f"  {key}: max_diff={diff.max():.6f} "
                          f"corr={corr:.6f} scale={scale:.6f}")
                else:
                    print(f"  {key}: shape mismatch HF={hf_val.shape} Metal={metal_val.shape}")
            else:
                print(f"  {key}: no HF reference")

    print("\nDone.")


if __name__ == "__main__":
    main()
