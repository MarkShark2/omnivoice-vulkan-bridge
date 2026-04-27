#!/usr/bin/env python3
"""
Step 1 of the KV-cache re-export: prove we can
  (a) load the k2-fsa/OmniVoice PyTorch checkpoint on CPU,
  (b) run a forward pass with a custom 4D bidirectional attention mask,
  (c) run the SAME inputs through the underlying Qwen3Model twice —
      once monolithically and once split into (prefix pass) + (step pass
      with past_key_values) — and compare numerics.

The whole point is to find out, BEFORE touching ONNX or FP16 anything,
whether the HF Qwen3 implementation in transformers 5.5 actually supports
the combination:
    - 4D boolean attention mask, custom pattern (bidirectional prefix-LM-ish)
    - past_key_values from a previous call
    - custom position_ids that continue the sequence
If those three things compose correctly on CPU in fp32 to within a small
numeric tolerance of a monolithic forward, then the approximate-prefix KV
cache plan is viable and we can proceed to the wrapper + ONNX export.

We intentionally test the two-pass split in APPROXIMATION mode:
  - Pass 1 (prefix-only): input_ids[:, :S_p], attention_mask[:S_p,:S_p]
  - Pass 2 (target, with past KV from pass 1): input_ids[:, S_p:],
    attention_mask[S_p:, :S_total]
The expected numeric difference vs the monolithic forward is NON-ZERO when
the full model uses bidirectional attention (prefix sees target in the
original). That's the "approximation" cost. We just want to:
  (i)  confirm the API plumbs correctly and shapes line up, and
  (ii) measure the per-step logit deviation so we know if it's reasonable.
"""
import os
import sys
import time
from pathlib import Path

import torch

# Keep CPU threading modest; we're only doing one forward.
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

# The k2-fsa checkpoint dir. We hand the snapshot path directly to
# from_pretrained so huggingface_hub doesn't try to talk to the network.
CACHE_ROOT = Path("/home/mark/omnivoice/models--k2-fsa--OmniVoice")


def resolve_snapshot(cache_root: Path) -> Path:
    """Pick the HF snapshot directory that actually has config.json."""
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


def main():
    # Make the in-repo OmniVoice package importable without installing it.
    sys.path.insert(0, "/home/mark/omnivoice/OmniVoice")
    # The OmniVoice package has optional deps (torchaudio, duration estimator,
    # text utils) that import unconditionally. All needed for from_pretrained.
    from omnivoice.models.omnivoice import OmniVoice

    snap = resolve_snapshot(CACHE_ROOT)
    print(f"[sanity] snapshot: {snap}")

    # train=True skips the audio_tokenizer / ASR / duration estimator side-load.
    # We don't need any of those for pure LLM+audio_heads forward testing.
    print("[sanity] loading OmniVoice (CPU, fp32, no side-load)...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        str(snap),
        train=True,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    print(f"[sanity]   loaded in {time.time() - t0:.1f}s")

    cfg = model.config
    llm_cfg = cfg.llm_config
    print(
        f"[sanity] cfg: layers={llm_cfg.num_hidden_layers} "
        f"heads={llm_cfg.num_attention_heads} kv_heads={llm_cfg.num_key_value_heads} "
        f"head_dim={llm_cfg.head_dim} hidden={llm_cfg.hidden_size} "
        f"audio_codebooks={cfg.num_audio_codebook} audio_vocab={cfg.audio_vocab_size}"
    )

    # --- Build a small synthetic input that matches the diffusion-time layout.
    # Shapes: input_ids (B, C, S), audio_mask (B, S), attention_mask (B, 1, S, S).
    B = 1
    C = cfg.num_audio_codebook  # 8
    S_prefix = 24  # small prefix (text tokens only, no audio mask)
    S_target = 16  # small target (all-mask audio slots)
    S = S_prefix + S_target

    torch.manual_seed(0)
    # Prefix: random text tokens; audio_mask=False so they're text embeds.
    prefix_text_ids = torch.randint(
        low=0, high=llm_cfg.vocab_size, size=(B, 1, S_prefix), dtype=torch.long
    )
    # Replicate across codebook dim just to fill the tensor; audio_mask=False
    # means only codebook 0 is actually read by _prepare_embed_inputs as text.
    prefix_ids_full = prefix_text_ids.expand(B, C, S_prefix).contiguous()

    # Target: all audio_mask positions filled with the audio mask id (1024).
    target_ids = torch.full(
        (B, C, S_target), cfg.audio_mask_id, dtype=torch.long
    )

    input_ids = torch.cat([prefix_ids_full, target_ids], dim=-1)  # (B, C, S)

    audio_mask = torch.zeros((B, S), dtype=torch.bool)
    audio_mask[:, S_prefix:] = True  # target positions are audio

    # Bidirectional full-attention mask over the whole sequence (as OmniVoice does).
    attn_full = torch.ones((B, 1, S, S), dtype=torch.bool)

    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, S)

    # --- (a) Monolithic forward (reference)
    print("[sanity] monolithic forward...")
    t0 = time.time()
    with torch.inference_mode():
        out_full = model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=attn_full,
            position_ids=position_ids,
        )
    logits_full = out_full.logits  # (B, C, S, V)
    print(f"[sanity]   done in {time.time() - t0:.2f}s logits={tuple(logits_full.shape)}")

    # --- (b) Two-pass: prefix then target with past_key_values.
    # We hook into model.llm directly so we can keep the KV cache ourselves.
    llm = model.llm

    # Prepare embeds the same way OmniVoice.forward does (so the two paths
    # differ only in how we feed them through the LLM).
    with torch.inference_mode():
        inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)

    prefix_embeds = inputs_embeds[:, :S_prefix, :]
    target_embeds = inputs_embeds[:, S_prefix:, :]

    # Prefix attention: prefix positions only see prefix.
    attn_prefix = torch.ones((B, 1, S_prefix, S_prefix), dtype=torch.bool)
    pos_prefix = torch.arange(S_prefix, dtype=torch.long).unsqueeze(0)

    print("[sanity] pass 1: prefix-only forward through llm...")
    t0 = time.time()
    with torch.inference_mode():
        llm_out_p = llm(
            inputs_embeds=prefix_embeds,
            attention_mask=attn_prefix,
            position_ids=pos_prefix,
            use_cache=True,
            return_dict=True,
        )
    past_kv = llm_out_p.past_key_values
    print(f"[sanity]   done in {time.time() - t0:.2f}s "
          f"past_kv type={type(past_kv).__name__}")

    # Target attention: S_target rows, each attending to all S_prefix+S_target cols.
    attn_tgt = torch.ones((B, 1, S_target, S), dtype=torch.bool)
    pos_tgt = torch.arange(S_prefix, S, dtype=torch.long).unsqueeze(0)

    print("[sanity] pass 2: target forward with past_key_values...")
    t0 = time.time()
    with torch.inference_mode():
        llm_out_t = llm(
            inputs_embeds=target_embeds,
            attention_mask=attn_tgt,
            position_ids=pos_tgt,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
    hidden_tgt = llm_out_t.last_hidden_state  # (B, S_target, H)
    print(f"[sanity]   done in {time.time() - t0:.2f}s hidden={tuple(hidden_tgt.shape)}")

    # Convert hidden_tgt through audio_heads the same way OmniVoice.forward does.
    with torch.inference_mode():
        logits_tgt = model.audio_heads(hidden_tgt)
        logits_tgt = logits_tgt.view(B, S_target, C, cfg.audio_vocab_size).permute(0, 2, 1, 3)

    # --- (c) Compare target-region logits between monolithic and two-pass.
    ref_tgt_logits = logits_full[:, :, S_prefix:, :]
    diff = (ref_tgt_logits - logits_tgt).abs()
    print(
        f"[sanity] target logits |mono - split|: "
        f"max={diff.max().item():.4g}  mean={diff.mean().item():.4g}  "
        f"ref |mean|={ref_tgt_logits.abs().mean().item():.4g}  "
        f"argmax_top1_agree={(ref_tgt_logits.argmax(-1) == logits_tgt.argmax(-1)).float().mean().item()*100:.1f}%"
    )

    # Exit non-zero if the API fundamentally broke (shape mismatch would raise
    # above already; here we just flag absurd values that would indicate a
    # silent wiring bug).
    if not torch.isfinite(logits_tgt).all():
        print("[sanity] FAIL: non-finite logits in split forward")
        sys.exit(2)

    print("[sanity] OK")


if __name__ == "__main__":
    main()
