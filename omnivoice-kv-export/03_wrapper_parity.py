#!/usr/bin/env python3
"""
Validate the OmniVoiceKvWrapper against the monolithic OmniVoice forward,
in two scenarios that mirror how the ONNX graph will actually be driven:

  1. PREFIX MODE (step 0 of the diffusion loop)
     past_k/past_v are arbitrary (we pass zeros); target_positions selects
     every position; S_new == S_full. Scatter writes every position, so
     present_kv == new_kv == what a fresh monolithic forward would produce.
     Expected: logits match the monolithic forward BIT-EXACTLY (no approx).

  2. STEP MODE (step 1..N-1)
     past_k/past_v = the *true* monolithic prefix K/V (captured from a
     regular forward with use_cache=True on the full sequence).
     target_positions = per-row absolute positions of the target region.
     Input is just the target tokens.
     Expected: target logits match monolithic target logits (up to the
     scatter being equivalent to the cache already holding those values —
     which should also be BIT-EXACT, since the target K/V in the
     monolithic cache is exactly what our scatter overwrites with).

If (1) is exact, the pure-PyTorch wiring is correct end-to-end.
If (2) is exact against the monolithic cache, scatter semantics are correct.
The *approximation* we'll actually pay at inference time kicks in only when
past_kv is the frozen step-0 cache (different from the dynamically correct
monolithic one). We already measured that gap separately in
02_kv_plumbing.py.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from paths import OMNIVOICE_SRC, resolve_hf_snapshot

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
torch.set_grad_enabled(False)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(OMNIVOICE_SRC))


def main():
    from omnivoice.models.omnivoice import OmniVoice
    from kv_wrapper import OmniVoiceKvWrapper

    snap = resolve_hf_snapshot()
    print(f"[parity] snapshot: {snap}")

    model = OmniVoice.from_pretrained(
        str(snap), train=True, dtype=torch.float32, device_map="cpu"
    ).eval()
    # Force eager attention so we export + validate against the same path.
    model.llm.config._attn_implementation = "eager"

    cfg = model.config
    llm_cfg = cfg.llm_config
    L = llm_cfg.num_hidden_layers
    kv_h = llm_cfg.num_key_value_heads
    d = llm_cfg.head_dim

    wrapper = OmniVoiceKvWrapper(model).eval()

    # --- Build a batched input that mirrors the (cond, uncond) layout
    # currently used by OmniVoice.generate: both rows padded to maxLen, with
    # different active-region lengths. Target audio mask slot(s) live at the
    # END of each row's active region.
    B = 2
    C = cfg.num_audio_codebook
    c_len = 40  # cond active length
    u_len = 24  # uncond active length
    S_target = 12
    maxLen = c_len
    c_prefix = c_len - S_target  # 28
    u_prefix = u_len - S_target  # 12

    torch.manual_seed(0)

    # Row 0 (cond): text prefix + all-mask target, then padding out to maxLen.
    ids_cond = torch.full((1, C, maxLen), cfg.audio_mask_id, dtype=torch.long)
    # Random text token ids for the prefix; only codebook 0 is read as text when
    # audio_mask is False. Fill all codebooks with arbitrary values.
    ids_cond[0, :, :c_prefix] = torch.randint(
        0, llm_cfg.vocab_size, (C, c_prefix), dtype=torch.long
    )
    ids_cond[0, :, c_prefix:c_len] = cfg.audio_mask_id  # target = all mask

    # Row 1 (uncond): tiny text-ish prefix + target, padding out to maxLen.
    ids_uncond = torch.full((1, C, maxLen), cfg.audio_mask_id, dtype=torch.long)
    ids_uncond[0, :, :u_prefix] = torch.randint(
        0, llm_cfg.vocab_size, (C, u_prefix), dtype=torch.long
    )
    ids_uncond[0, :, u_prefix:u_len] = cfg.audio_mask_id

    input_ids = torch.cat([ids_cond, ids_uncond], dim=0)  # (2, C, maxLen)

    audio_mask = torch.zeros((B, maxLen), dtype=torch.bool)
    audio_mask[0, c_prefix:c_len] = True  # cond target slots are audio
    audio_mask[1, u_prefix:u_len] = True  # uncond target slots are audio

    # Bidirectional mask replicating what OmniVoice.generate builds:
    #   cond row:   [0..c_len) active (full block), rest all False
    #   uncond row: [0..u_len) active, diagonal-only for pad positions
    attn_full = torch.zeros((B, 1, maxLen, maxLen), dtype=torch.bool)
    attn_full[0, :, :c_len, :c_len] = True
    attn_full[1, :, :u_len, :u_len] = True
    for p in range(u_len, maxLen):
        attn_full[1, :, p, p] = True

    position_ids = torch.arange(maxLen).unsqueeze(0).expand(B, maxLen).contiguous()

    # -------- Reference: monolithic forward (with use_cache=True to snapshot
    # the "true" per-layer K/V for Scenario 2).
    llm_out = model.llm(
        inputs_embeds=model._prepare_embed_inputs(input_ids, audio_mask),
        attention_mask=attn_full,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    ref_hidden = llm_out.last_hidden_state  # (B, maxLen, H)
    ref_cache = llm_out.past_key_values
    ref_logits = (
        model.audio_heads(ref_hidden)
        .view(B, maxLen, C, cfg.audio_vocab_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    # ========== Scenario 1: PREFIX MODE ==========
    # target_positions = [0..maxLen-1] broadcast: scatter overwrites everything.
    tp_prefix = torch.arange(maxLen).unsqueeze(0).expand(B, maxLen).contiguous()

    past_zeros_k = [torch.zeros(B, kv_h, maxLen, d) for _ in range(L)]
    past_zeros_v = [torch.zeros(B, kv_h, maxLen, d) for _ in range(L)]

    out1 = wrapper(
        input_ids, audio_mask, attn_full, position_ids, tp_prefix,
        *past_zeros_k, *past_zeros_v,
    )
    logits1 = out1[0]
    present_k = list(out1[1 : 1 + L])
    present_v = list(out1[1 + L :])

    diff1 = (logits1 - ref_logits).abs()
    print(
        f"[parity] scenario 1 (prefix mode): "
        f"max={diff1.max().item():.4g}  mean={diff1.mean().item():.4g}  "
        f"match={(diff1.max().item() < 1e-4)}"
    )

    # Also sanity-check present_kv matches the monolithic cache.
    for li in range(L):
        k_ref = ref_cache.layers[li].keys
        v_ref = ref_cache.layers[li].values
        dk = (present_k[li] - k_ref).abs().max().item()
        dv = (present_v[li] - v_ref).abs().max().item()
        if dk > 1e-4 or dv > 1e-4:
            print(f"[parity]   layer {li}: K diff={dk:.4g} V diff={dv:.4g}")
            break
    else:
        print(f"[parity]   all {L} layers' present_kv match monolithic cache")

    # ========== Scenario 2: STEP MODE with ORACLE past_kv ==========
    # Feed the true monolithic K/V as past. Pass only target tokens as input.
    # Expected: target logits match monolithic target logits bit-exactly
    # (because scatter overwrites the target slots with the same values the
    # monolithic cache already held there — same computation, same result).
    past_k_oracle = [ref_cache.layers[li].keys.clone() for li in range(L)]
    past_v_oracle = [ref_cache.layers[li].values.clone() for li in range(L)]

    # Target positions per batch row.
    tp_step = torch.stack(
        [torch.arange(c_prefix, c_len), torch.arange(u_prefix, u_len)], dim=0
    )  # (B, S_target)

    # Pull out target-only slices of input_ids, audio_mask, position_ids.
    # Because target offsets differ per row we build this manually.
    tgt_input_ids = torch.stack(
        [input_ids[0, :, c_prefix:c_len], input_ids[1, :, u_prefix:u_len]], dim=0
    )
    tgt_audio_mask = torch.stack(
        [audio_mask[0, c_prefix:c_len], audio_mask[1, u_prefix:u_len]], dim=0
    )
    tgt_position_ids = tp_step.clone()

    # Attention mask for step mode: target queries (length S_target) attend to
    # full-length K/V (maxLen). For row 0: queries at abs positions [c_prefix,c_len)
    # attend to keys [0..c_len). For row 1: queries at abs [u_prefix,u_len)
    # attend to keys [0..u_len).
    step_attn = torch.zeros((B, 1, S_target, maxLen), dtype=torch.bool)
    step_attn[0, :, :, :c_len] = True
    step_attn[1, :, :, :u_len] = True

    out2 = wrapper(
        tgt_input_ids, tgt_audio_mask, step_attn, tgt_position_ids, tp_step,
        *past_k_oracle, *past_v_oracle,
    )
    logits2 = out2[0]  # (B, C, S_target, V)

    # Compare to monolithic logits at target positions.
    ref_tgt_logits = torch.stack(
        [ref_logits[0, :, c_prefix:c_len, :], ref_logits[1, :, u_prefix:u_len, :]],
        dim=0,
    )
    diff2 = (logits2 - ref_tgt_logits).abs()
    print(
        f"[parity] scenario 2 (step mode, oracle past_kv): "
        f"max={diff2.max().item():.4g}  mean={diff2.mean().item():.4g}  "
        f"match={(diff2.max().item() < 1e-3)}"
    )

    if diff1.max().item() > 1e-4 or diff2.max().item() > 1e-3:
        print("[parity] FAIL")
        sys.exit(2)
    print("[parity] OK")


if __name__ == "__main__":
    main()
