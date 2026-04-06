#!/usr/bin/env python3
"""Dump reference intermediate values from Qwen3.5-0.8B GDN layer 0 (HuggingFace).

Usage:
    cd /Users/jacobfreck/Source/ironmill-qwen35
    python3 scripts/gdn_reference.py
"""

import torch
import json
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

MODEL_DIR = Path("models/Qwen3.5-0.8B")

def load_config():
    with open(MODEL_DIR / "config.json") as f:
        raw = json.load(f)
    # Qwen3.5 nests text model config under "text_config"
    return raw.get("text_config", raw)

def rms_norm(x, weight, eps=1e-6):
    """Qwen3.5 centered RMSNorm: y = x / rms * (1 + weight)."""
    var = (x * x).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)) * (weight + 1.0)

def main():
    cfg = load_config()
    h = cfg["hidden_size"]
    eps = cfg.get("rms_norm_eps", 1e-6)

    kd = cfg.get("linear_key_head_dim", 128)
    vd = cfg.get("linear_value_head_dim", 128)
    nkh = cfg.get("linear_num_key_heads", 16)
    nvh = cfg.get("linear_num_value_heads", 16)
    conv_k = cfg.get("linear_conv_kernel_dim", 4)
    key_dim = kd * nkh   # 2048
    val_dim = vd * nvh    # 2048
    qkv_dim = key_dim + key_dim + val_dim  # 6144

    print(f"Config: h={h}, key_dim={key_dim}, val_dim={val_dim}, qkv_dim={qkv_dim}")
    print(f"  nkh={nkh}, nvh={nvh}, kd={kd}, vd={vd}, conv_k={conv_k}, eps={eps}")

    # Load weights
    st_files = sorted(MODEL_DIR.glob("*.safetensors"))
    tensors = {}
    for f in st_files:
        tensors.update(load_file(str(f)))

    # GDN layer 0 weights (note: model.language_model prefix, linear_attn not self_attn)
    pfx = "model.language_model.layers.0"
    qkv_w = tensors[f"{pfx}.linear_attn.in_proj_qkv.weight"].float()  # [qkv_dim, h]
    z_w   = tensors[f"{pfx}.linear_attn.in_proj_z.weight"].float()    # [val_dim, h]
    a_proj_w = tensors[f"{pfx}.linear_attn.in_proj_a.weight"].float() # [nkh, h]
    b_proj_w = tensors[f"{pfx}.linear_attn.in_proj_b.weight"].float() # [nkh, h]
    a_log  = tensors[f"{pfx}.linear_attn.A_log"].float()              # [nkh]
    dt_bias_w = tensors[f"{pfx}.linear_attn.dt_bias"].float()         # [nkh]  (may be bf16)
    if f"{pfx}.linear_attn.dt_bias" in tensors:
        dt_bias_w = tensors[f"{pfx}.linear_attn.dt_bias"].float()
    norm_w = tensors[f"{pfx}.linear_attn.norm.weight"].float()        # [v_head_dim]
    out_w  = tensors[f"{pfx}.linear_attn.out_proj.weight"].float()    # [h, val_dim]
    conv_w = tensors[f"{pfx}.linear_attn.conv1d.weight"].float()      # [qkv_dim, 1, conv_k]
    post_norm_w = tensors[f"{pfx}.post_attention_layernorm.weight"].float()

    embed = tensors["model.language_model.embed_tokens.weight"].float()
    inp_norm_w = tensors[f"{pfx}.input_layernorm.weight"].float()

    # Check shapes
    print(f"\nWeight shapes:")
    print(f"  embed:     {embed.shape}")
    print(f"  inp_norm:  {inp_norm_w.shape}")
    print(f"  qkv_proj:  {qkv_w.shape}")
    print(f"  z_proj:    {z_w.shape}")
    print(f"  a_proj:    {a_proj_w.shape}")
    print(f"  b_proj:    {b_proj_w.shape}")
    print(f"  a_log:     {a_log.shape}")
    print(f"  dt_bias:   {dt_bias_w.shape}")
    print(f"  norm_w:    {norm_w.shape}")
    print(f"  conv1d_w:  {conv_w.shape}")
    print(f"  o_proj:    {out_w.shape}")
    print(f"  post_norm: {post_norm_w.shape}")

    # --- Forward pass (prefill, first 4 tokens from wikitext2) ---
    import json as json2
    with open("tests/fixtures/quality/wikitext2-qwen35.json") as f2:
        wt = json2.load(f2)
    # Use the same tokens the GPU debug shows (first pipeline call)
    token_ids = torch.tensor([9707, 1879], dtype=torch.long)
    T = len(token_ids)
    print(f"\nToken IDs: {token_ids.tolist()}")

    # Step 1: Embedding
    x = embed[token_ids]  # [T, h]
    print(f"\n[REF] Embedding (token 0, first 5): {x[0, :5].tolist()}")

    # Step 2: Input RMSNorm for layer 0
    normed = rms_norm(x, inp_norm_w, eps)
    print(f"[REF] After input norm (token 0, first 5): {normed[0, :5].tolist()}")

    # Step 3: QKV projection
    qkv = normed @ qkv_w.T  # [T, qkv_dim]
    print(f"[REF] QKV proj (token 0, first 5): {qkv[0, :5].tolist()}")

    # Split into Q, K, V
    q = qkv[:, :key_dim]            # [T, key_dim]
    k = qkv[:, key_dim:2*key_dim]   # [T, key_dim]
    v = qkv[:, 2*key_dim:]          # [T, val_dim]
    print(f"[REF] Q (token 0, first 5): {q[0, :5].tolist()}")
    print(f"[REF] K (token 0, first 5): {k[0, :5].tolist()}")
    print(f"[REF] V (token 0, first 5): {v[0, :5].tolist()}")

    # Step 4: Z projection (gate)
    z = normed @ z_w.T  # [T, val_dim]
    print(f"[REF] Z proj (token 0, first 5): {z[0, :5].tolist()}")

    # Step 4b: A and B projections (input-dependent gates)
    a_proj = normed @ a_proj_w.T  # [T, nkh]
    b_proj = normed @ b_proj_w.T  # [T, nkh]
    print(f"[REF] A proj (token 0): {a_proj[0, :5].tolist()}")
    print(f"[REF] B proj (token 0): {b_proj[0, :5].tolist()}")

    # Step 5: Conv1d + SiLU on FULL QKV (6144 channels)
    # conv_w shape: [6144, 1, 4] (depthwise conv1d)
    conv_state = torch.zeros(conv_k - 1, qkv_dim)  # [3, 6144] for padding
    qkv_flat = qkv  # [T, qkv_dim]
    
    conv_out = torch.zeros(T, qkv_dim)
    for t in range(T):
        for ch in range(qkv_dim):
            val = 0.0
            # state holds [t-3, t-2, t-1] for this channel
            for j in range(conv_k - 1):
                val += conv_w[ch, 0, j].item() * conv_state[j, ch].item()
            val += conv_w[ch, 0, conv_k - 1].item() * qkv_flat[t, ch].item()
            conv_out[t, ch] = val
        # shift state left, append current input
        for j in range(conv_k - 2):
            conv_state[j] = conv_state[j + 1]
        conv_state[conv_k - 2] = qkv_flat[t]
    
    # SiLU activation on conv output
    conv_silu = conv_out * torch.sigmoid(conv_out)
    print(f"[REF] Conv+SiLU out (token 0, first 5 = Q head0): {conv_silu[0, :5].tolist()}")
    print(f"[REF] Conv+SiLU out (token 0, K head0 first 5): {conv_silu[0, key_dim:key_dim+5].tolist()}")
    print(f"[REF] Conv+SiLU out (token 0, V head0 first 5): {conv_silu[0, 2*key_dim:2*key_dim+5].tolist()}")

    # Step 6: Recurrent update (delta rule per head)
    print(f"[REF] a_log raw: {a_log[:5].tolist()}")
    print(f"[REF] dt_bias: {dt_bias_w[:5].tolist()}")

    # Recurrent state: [nkh, vd, kd] (S_h) — float32
    S = torch.zeros(nkh, vd, kd)  # state per head

    outputs = torch.zeros(T, val_dim)  # recurrent output

    for t in range(T):
        for head in range(nkh):
            q_h = conv_silu[t, head * kd:(head + 1) * kd]           # [kd] from Q section
            k_h = conv_silu[t, key_dim + head * kd:key_dim + (head + 1) * kd]  # [kd] from K section
            v_h = conv_silu[t, 2 * key_dim + head * vd:2 * key_dim + (head + 1) * vd]  # [vd] from V section

            # L2 normalize Q and K
            q_norm = torch.norm(q_h)
            k_norm = torch.norm(k_h)
            q_n = q_h / (q_norm + 1e-6) if q_norm > 0 else q_h
            k_n = k_h / (k_norm + 1e-6) if k_norm > 0 else k_h

            # Query scaling
            scale = 1.0 / (kd ** 0.5)

            # Gates: beta = sigmoid(b_proj), dt = softplus(a_proj + dt_bias), decay = exp(-exp(A_log) * dt)
            b_val = b_proj[t, head].item()
            beta = 1.0 / (1.0 + np.exp(-b_val))  # sigmoid
            
            a_val = a_proj[t, head].item() + dt_bias_w[head].item()
            dt = np.log(1.0 + np.exp(a_val)) if a_val <= 20 else a_val  # softplus
            decay = np.exp(-np.exp(a_log[head].item()) * dt)

            # Delta rule: memory read
            kv_mem = S[head] @ k_n  # [vd] — retrieval using current state

            # State update with decay
            for vi in range(vd):
                delta_vi = beta * (v_h[vi].item() - kv_mem[vi].item())
                o_sum = 0.0
                for ki in range(kd):
                    s_decayed = decay * S[head, vi, ki].item()
                    s_new = s_decayed + k_n[ki].item() * delta_vi
                    S[head, vi, ki] = s_new
                    o_sum += s_new * q_n[ki].item() * scale
                outputs[t, head * vd + vi] = o_sum

    print(f"[REF] Recurrent out (token 0, first 5): {outputs[0, :5].tolist()}")
    print(f"[REF] Recurrent out (token 1, first 5): {outputs[1, :5].tolist()}")

    # Step 7: Output gate: RMSNorm(recurrent_out) * silu(z) per head
    gated = torch.zeros(T, val_dim)
    for t in range(T):
        for head in range(nvh):
            sl = slice(head * vd, (head + 1) * vd)
            o = outputs[t, sl]
            # Per-head RMSNorm (no centering, uses norm_weight)
            rms = torch.sqrt((o * o).mean() + eps)
            normed_o = o / rms * norm_w   # norm_w is [vd]
            # silu gate
            z_h = z[t, sl]
            gate = z_h * torch.sigmoid(z_h)
            gated[t, sl] = normed_o * gate

    print(f"[REF] Gated output (token 0, first 5): {gated[0, :5].tolist()}")

    # Step 8: Output projection
    attn_out = gated @ out_w.T  # [T, h]
    print(f"[REF] Out proj (token 0, first 5): {attn_out[0, :5].tolist()}")

    # Step 9: Residual + post-attention norm
    residual = x + attn_out  # [T, h]
    post_normed = rms_norm(residual, post_norm_w, eps)
    print(f"[REF] Post-attn norm (token 0, first 5): {post_normed[0, :5].tolist()}")
    print(f"[REF] Post-attn norm (token {T-1}, first 5): {post_normed[-1, :5].tolist()}")

if __name__ == "__main__":
    main()
