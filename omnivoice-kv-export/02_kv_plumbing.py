#!/usr/bin/env python3
"""
Isolate "API plumbing" from "approximation cost" in the KV-cache plan.

We run the SAME inputs three ways:
  (A) Monolithic forward (reference).
  (B) Prefix-only forward, then target forward using (B)'s past_kv.
      → differs from (A) because prefix never saw the target (this is the
        approximation we'll ship).
  (C) Monolithic forward with use_cache=True, then target forward using
      a DynamicCache populated from (A)'s full-sequence K/V sliced to the
      prefix region.
      → if the HF past_key_values API is wired correctly, this should
        exactly reproduce (A)'s target logits.

If (C) ≈ (A), plumbing is proven correct and the (B)-vs-(A) gap is
purely the bidirectional-prefix approximation (what we're intentionally
trading for speed). If (C) ≠ (A), something is wrong with how we're
plumbing the cache and we need to fix that before exporting.

We also dump a few real-world-ish stats (softmax-distribution distance)
so we have a better sense than argmax-top1-agreement on random inputs,
which is too noisy to be meaningful.
"""
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from paths import OMNIVOICE_CACHE_ROOT, OMNIVOICE_SRC

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
# Use no_grad instead of inference_mode: we need to do tensor-slice/update
# operations on the cached K/V outside the original forward, and inference-mode
# tensors can't be fed back into autograd-tracked code (e.g. .view + linear).
torch.set_grad_enabled(False)
CACHE_ROOT = OMNIVOICE_CACHE_ROOT


def resolve_snapshot(cache_root: Path) -> Path:
    refs = cache_root / "refs" / "main"
    snapshots = cache_root / "snapshots"
    if refs.is_file():
        commit = refs.read_text().strip()
        cand = snapshots / commit
        if (cand / "config.json").is_file():
            return cand
    for entry in sorted(snapshots.iterdir()):
        if (entry / "config.json").is_file():
            return entry
    raise FileNotFoundError(f"No snapshot with config.json under {cache_root}")


def softmax_js_div(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> float:
    """Jensen–Shannon divergence between per-position softmax distributions."""
    pa = F.softmax(a.float(), dim=dim)
    pb = F.softmax(b.float(), dim=dim)
    m = 0.5 * (pa + pb)
    eps = 1e-12
    kl_a = (pa * (pa.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim)
    kl_b = (pb * (pb.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim)
    return (0.5 * (kl_a + kl_b)).mean().item()


def main():
    sys.path.insert(0, str(OMNIVOICE_SRC))
    from omnivoice.models.omnivoice import OmniVoice
    from transformers import DynamicCache

    snap = resolve_snapshot(CACHE_ROOT)
    print(f"[plumb] snapshot: {snap}")

    model = OmniVoice.from_pretrained(
        str(snap), train=True, dtype=torch.float32, device_map="cpu"
    ).eval()

    cfg = model.config
    llm = model.llm
    llm_cfg = cfg.llm_config

    B = 1
    C = cfg.num_audio_codebook
    S_prefix = 24
    S_target = 16
    S = S_prefix + S_target

    torch.manual_seed(0)
    prefix_ids = torch.randint(0, llm_cfg.vocab_size, (B, 1, S_prefix), dtype=torch.long)
    prefix_ids_full = prefix_ids.expand(B, C, S_prefix).contiguous()
    target_ids = torch.full((B, C, S_target), cfg.audio_mask_id, dtype=torch.long)
    input_ids = torch.cat([prefix_ids_full, target_ids], dim=-1)
    audio_mask = torch.zeros((B, S), dtype=torch.bool)
    audio_mask[:, S_prefix:] = True
    attn_full = torch.ones((B, 1, S, S), dtype=torch.bool)
    pos_full = torch.arange(S, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        embeds = model._prepare_embed_inputs(input_ids, audio_mask)

    # ----- (A) Monolithic with use_cache=True so we can snapshot true K/V -----
    with torch.no_grad():
        llm_out_A = llm(
            inputs_embeds=embeds,
            attention_mask=attn_full,
            position_ids=pos_full,
            use_cache=True,
            return_dict=True,
        )
    hidden_A = llm_out_A.last_hidden_state  # (B, S, H)
    full_past = llm_out_A.past_key_values  # filled by the monolithic pass
    logits_A = model.audio_heads(hidden_A).view(B, S, C, cfg.audio_vocab_size).permute(0, 2, 1, 3)

    # Sanity: at this point `full_past` has K/V for positions 0..S-1, and these
    # are the TRUE ones from the bidirectional pass (including target).
    print(f"[plumb] monolithic cache layers: {len(full_past.layers)}")
    k0, v0 = full_past.layers[0].keys, full_past.layers[0].values
    print(f"[plumb] layer-0 K shape={tuple(k0.shape)} V shape={tuple(v0.shape)}")

    # ----- (B) Prefix-only then target-with-past_kv (the APPROXIMATION) -----
    attn_pref = torch.ones((B, 1, S_prefix, S_prefix), dtype=torch.bool)
    pos_pref = torch.arange(S_prefix, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        llm_out_B1 = llm(
            inputs_embeds=embeds[:, :S_prefix],
            attention_mask=attn_pref,
            position_ids=pos_pref,
            use_cache=True,
            return_dict=True,
        )
    past_B = llm_out_B1.past_key_values
    attn_tgt = torch.ones((B, 1, S_target, S), dtype=torch.bool)
    pos_tgt = torch.arange(S_prefix, S, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        llm_out_B2 = llm(
            inputs_embeds=embeds[:, S_prefix:],
            attention_mask=attn_tgt,
            position_ids=pos_tgt,
            past_key_values=past_B,
            use_cache=True,
            return_dict=True,
        )
    hidden_B_tgt = llm_out_B2.last_hidden_state
    logits_B_tgt = (
        model.audio_heads(hidden_B_tgt)
        .view(B, S_target, C, cfg.audio_vocab_size)
        .permute(0, 2, 1, 3)
    )

    # ----- (C) Oracle: feed the TRUE monolithic prefix K/V as past_kv -----
    # Build a fresh DynamicCache whose per-layer (K, V) are (B, kv_h, S_prefix, head_dim)
    # taken from the monolithic cache. If the HF cache API honours these
    # as "already computed up to S_prefix", the target forward should reproduce
    # the monolithic target logits exactly.
    oracle_cache = DynamicCache(config=llm_cfg)
    num_layers = llm_cfg.num_hidden_layers
    for li in range(num_layers):
        k_full = full_past.layers[li].keys
        v_full = full_past.layers[li].values
        k_pref = k_full[:, :, :S_prefix, :].contiguous()
        v_pref = v_full[:, :, :S_prefix, :].contiguous()
        oracle_cache.update(k_pref, v_pref, li)

    with torch.no_grad():
        llm_out_C = llm(
            inputs_embeds=embeds[:, S_prefix:],
            attention_mask=attn_tgt,
            position_ids=pos_tgt,
            past_key_values=oracle_cache,
            use_cache=True,
            return_dict=True,
        )
    hidden_C_tgt = llm_out_C.last_hidden_state
    logits_C_tgt = (
        model.audio_heads(hidden_C_tgt)
        .view(B, S_target, C, cfg.audio_vocab_size)
        .permute(0, 2, 1, 3)
    )

    # ----- Compare -----
    ref_tgt = logits_A[:, :, S_prefix:, :]
    for name, logits_x in [("B: prefix-only approx", logits_B_tgt),
                            ("C: oracle cache      ", logits_C_tgt)]:
        diff = (ref_tgt - logits_x).abs()
        rel = diff.mean().item() / ref_tgt.abs().mean().item()
        js = softmax_js_div(ref_tgt, logits_x)
        top1 = (ref_tgt.argmax(-1) == logits_x.argmax(-1)).float().mean().item() * 100
        print(
            f"[plumb] {name}: max={diff.max().item():.4g} mean={diff.mean().item():.4g} "
            f"rel={rel*100:.2f}%  JS={js:.4g}  top1={top1:.1f}%"
        )


if __name__ == "__main__":
    main()
