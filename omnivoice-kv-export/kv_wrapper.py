#!/usr/bin/env python3
"""
Wrapper + custom cache that turns OmniVoice's iterative diffusion loop into a
single ONNX graph that supports a *stateful* K/V cache across steps, while
correctly handling the per-batch-row target offsets that the existing
(cond, uncond) batching uses.

Why scatter-update, not append
------------------------------
A typical causal LM KV cache uses APPEND semantics: new K/V tokens are
concatenated on the end of past K/V. That works because prefix tokens are
always strictly before target tokens in position order.

OmniVoice's batch layout doesn't cooperate with that model: both batch rows
run in parallel with DIFFERENT prefix lengths (cond has the ref audio tokens
+ text; uncond has just text). The target audio tokens therefore sit at
different absolute positions in each row's padded sequence. An append cache
would pick one layout and break the other.

Scatter-update handles it cleanly: the cache has a fixed length `S_full` per
utterance (the same as the current monolithic graph's padded `maxLen`), and
each step writes new target K/V at per-batch-row `target_positions` in that
buffer.

Why Slice+Concat, not torch.scatter
-----------------------------------
Earlier versions of this wrapper used `torch.scatter(-2, idx, src)`, which
torch-onnx emits as `ScatterElements`. That op turned out to be a problem on
onnxruntime-web's WebGPU EP: missing kernel in 1.20, numerically broken in
1.24+ for our (fp16, axis=-2) shape. The current implementation expresses the
SAME scatter-update semantics as a per-batch Slice+Concat:

    present[b] = cat( past[b, :, :off_b],  new[b],  past[b, :, off_b+S_new:] )

stacked over B=2. Each row's `off_b` is a dynamic tensor scalar pulled from
`target_positions[b, 0]`, and the range must be contiguous (which the driver
always produces). ONNX Slice + Concat kernels are available and correct on
every ORT-web EP we care about.

Both "prefix" and "step" calls use the same graph:

    step 0 ("prefix"):
        past_k, past_v  : any contents (we overwrite them all)
        target_positions: [0..S_full-1] broadcast across batch
        S_new           = S_full
        → scatter ends up writing every position → present_kv = new_kv

    step 1..N-1 ("step"):
        past_k, past_v  : the present_kv from step 0 (carried across steps)
        target_positions: per-row absolute indices of the target audio tokens
        S_new           = S_target (<< S_full)
        → scatter overwrites only target positions; prefix K/V stays frozen

Approximation boundary
----------------------
In step 1..N-1, the non-target positions of present_kv are the values they
had at step 0, when target was all-mask. Under bidirectional attention those
values theoretically depend on target content, so they drift across steps
in the original monolithic model. Freezing them is the approximation we're
gambling on.

This module is PURE PyTorch; it exports cleanly with torch.onnx.export. The
only trick is that Qwen3's attention expects a `past_key_values` object
whose `.update(k, v, layer_idx) -> (full_k, full_v)` returns the full
concatenated K/V. We supply a minimal duck-typed class (ScatterCache) that
does exactly that — but with scatter semantics instead of cat.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SliceConcatCache:
    """Minimal duck-typed replacement for transformers.DynamicCache.

    Only implements what Qwen3Attention.forward() and Qwen3Model.forward()
    actually call when given a 4D boolean attention_mask + explicit
    position_ids: `.update(key, value, layer_idx)` and `.get_seq_length()`.

    Semantics: scatter-update (same as before) — the cache has a fixed length
    S_full, and new K/V tokens overwrite target positions. The IMPLEMENTATION,
    however, is pure Slice+Concat per batch row instead of `torch.scatter`:

        present[b] = cat( past[b, :, :off_b],  new[b],  past[b, :, off_b+S_new:] )

    Why: onnxruntime-web's WebGPU kernel for ScatterElements on our
    (fp16, axis=-2, large index) shape was either unsupported (1.20) or
    numerically broken (1.24+) on AMD BC-250. Slice+Concat compiles to
    well-supported kernels in every ORT-web version we care about.

    The B=2 per-batch unroll is necessary because cond and uncond have
    DIFFERENT target offsets (cond starts after the ref-audio prefix; uncond
    starts near 0). A single vectorized op can't express two contiguous
    windows at different offsets without a scatter.
    """

    def __init__(
        self,
        past_k_list: List[torch.Tensor],
        past_v_list: List[torch.Tensor],
        target_positions: torch.Tensor,
    ):
        # past_k_list[i]: (B, kv_heads, S_full, head_dim). Same for past_v_list.
        # target_positions: (B, S_new) int64 — per-batch-row destination indices
        # in the K/V cache for the new tokens. REQUIRED: each row must be a
        # contiguous range [off_b .. off_b + S_new - 1]. (This is always true
        # in OmniVoice's current driver; the cache is invalid otherwise.)
        self.past_k = past_k_list
        self.past_v = past_v_list
        self.target_positions = target_positions
        self.present_k: List[torch.Tensor] = [None] * len(past_k_list)
        self.present_v: List[torch.Tensor] = [None] * len(past_v_list)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # key_states, value_states: (B, kv_heads, S_new, head_dim)
        past_k = self.past_k[layer_idx]
        past_v = self.past_v[layer_idx]

        B = past_k.shape[0]
        S_new = key_states.shape[-2]

        # Per-batch-row Slice+Concat. B is static (=2) so this unrolls cleanly
        # in ONNX. off_b is a traced tensor scalar -> ONNX Slice with dynamic
        # end inputs (opset 10+).
        rows_k = []
        rows_v = []
        for b in range(B):
            off = self.target_positions[b, 0]  # 0-d int64 tensor
            end = off + S_new                  # 0-d int64 tensor

            # past slices: [b:b+1, :, :off] and [b:b+1, :, end:]
            # In PREFIX mode (off=0, end=S_full) both slices are empty;
            # ONNX Concat handles zero-length inputs correctly.
            left_k = past_k[b : b + 1, :, :off, :]
            right_k = past_k[b : b + 1, :, end:, :]
            row_k = torch.cat([left_k, key_states[b : b + 1], right_k], dim=-2)
            rows_k.append(row_k)

            left_v = past_v[b : b + 1, :, :off, :]
            right_v = past_v[b : b + 1, :, end:, :]
            row_v = torch.cat([left_v, value_states[b : b + 1], right_v], dim=-2)
            rows_v.append(row_v)

        present_k = torch.cat(rows_k, dim=0)
        present_v = torch.cat(rows_v, dim=0)
        self.present_k[layer_idx] = present_k
        self.present_v[layer_idx] = present_v
        return present_k, present_v

    # The methods below are defensive: some HF cache call sites touch them
    # even when use_cache is True and position_ids is explicit. We want
    # tracing to produce pure tensor ops, not Python control flow branching
    # on these values, so all return simple static python values.

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.past_k[0].shape[-2]

    @property
    def is_sliding(self):
        return [False] * len(self.past_k)

    @property
    def is_compileable(self):
        return False

    def __len__(self) -> int:
        return len(self.past_k)

    # transformers sometimes does `if cache is not None and len(cache) == 0`
    # to detect an uninitialized cache. Our cache is always "initialized"
    # since past_k/past_v are provided up front.


class OmniVoiceKvWrapper(nn.Module):
    """nn.Module adapter around OmniVoice that accepts/returns explicit
    past/present K/V tensors, so the whole thing can be torch.onnx.export'd
    as a stateful step.

    Forward signature
    -----------------
        input_ids        (B, C, S_new)    int64
        audio_mask       (B, S_new)       bool
        attention_mask   (B, 1, S_new, S_full) bool
        position_ids     (B, S_new)       int64
        target_positions (B, S_new)       int64 — cache destination indices
        past_key_0..L-1  (B, kv_h, S_full, d) fp32  (L = num_hidden_layers)
        past_value_0..L-1 same

    Returns (as a tuple, to keep ONNX output names predictable):
        audio_logits     (B, C, S_new, V) fp32
        present_key_0..L-1   (B, kv_h, S_full, d) fp32
        present_value_0..L-1 same
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = model.config.llm_config.num_hidden_layers
        self.num_audio_codebook = model.config.num_audio_codebook
        self.audio_vocab_size = model.config.audio_vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        target_positions: torch.Tensor,
        *past_kv: torch.Tensor,
    ):
        L = self.num_layers
        assert len(past_kv) == 2 * L, (
            f"expected {2 * L} past_kv tensors, got {len(past_kv)}"
        )
        past_keys = list(past_kv[:L])
        past_values = list(past_kv[L:])

        cache = SliceConcatCache(past_keys, past_values, target_positions)

        inputs_embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)

        # use_cache=True is required so Qwen3Attention calls cache.update(...)
        # and thus populates cache.present_k/present_v for our outputs.
        llm_out = self.model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        hidden_states = llm_out.last_hidden_state  # (B, S_new, H)

        B, S_new, _ = hidden_states.shape
        C = self.num_audio_codebook
        V = self.audio_vocab_size

        logits_flat = self.model.audio_heads(hidden_states)
        # (B, S_new, C*V) → (B, C, S_new, V), matching the existing ONNX output shape.
        audio_logits = logits_flat.view(B, S_new, C, V).permute(0, 2, 1, 3).contiguous()

        # Return flat tuple so ONNX export assigns predictable output names.
        return (audio_logits, *cache.present_k, *cache.present_v)
