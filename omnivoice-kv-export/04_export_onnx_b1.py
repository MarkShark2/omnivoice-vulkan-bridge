#!/usr/bin/env python3
"""
Export OmniVoiceKvWrapper to ONNX **at batch size 1**.

Output: $OMNIVOICE_KV_B1_DIR/omnivoice-main-kv-b1.onnx
         + omnivoice-main-kv-b1.onnx_data

Why a B=1 variant
-----------------
The original export (04_export_onnx.py) traces at B=2 because OmniVoice's
monolithic graph always ran cond + uncond as one batched forward. That
worked, but has two problems for voice cloning:

  1. In step mode (steps 1..N-1), the uncond row's *effective* attention
     context is just its own S_new target tokens — its bAttn mask zeroes
     out everything past S_new. But ORT-Web 1.20's attention kernel still
     computes the full Q@K.T matrix over S_full keys before masking.
     With a 250-token ref-audio prefix, that's ~35% wasted attention
     compute per step.

  2. In prefix mode (step 0), the uncond row runs over S_full = maxLen
     even though it only uses its first uncondLen tokens. Same waste
     pattern, compounded by step 0 being the single most expensive step.

Running two B=1 forwards per step — cond with the full prefix, uncond
target-only — shrinks uncond's attention to S_new × S_new and lets the
cond call not pay for padding in the batch dimension either.

The exported graph is **identical in structure** to the B=2 variant
except that `SliceConcatCache.update`'s `for b in range(B)` unrolls to
one iteration instead of two. All other ops are batch-agnostic.

The JS side loads this graph once and calls it twice per diffusion step.
Weights load once (~1.2 GB fp16), so VRAM is unchanged vs the B=2 path.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from paths import B1_KV_DIR, OMNIVOICE_SRC, resolve_hf_snapshot

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
torch.set_grad_enabled(False)

HERE = Path(__file__).parent
OUT_DIR = B1_KV_DIR
OUT_ONNX = OUT_DIR / "omnivoice-main-kv-b1.onnx"
OUT_DATA_NAME = "omnivoice-main-kv-b1.onnx_data"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OMNIVOICE_SRC))


def _clear_output_files(keep: set[str] | None = None) -> None:
    keep = keep or set()
    if not OUT_DIR.is_dir():
        return
    for path in OUT_DIR.iterdir():
        if path.name in keep:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()


def _example_inputs_b1(model, S_full=24, S_new=None):
    """B=1 variant of 04_export_onnx.py's _example_inputs.

    We still trace in STEP MODE (S_new < S_full) so the scatter path is
    exercised and the unrolled Slice+Concat with a non-trivial target
    offset shows up in the traced graph. Prefix-mode and uncond-mode are
    just calls to the same graph with different shapes at runtime
    (S_new == S_full for prefix, S_new == S_full for uncond target-only).
    """
    cfg = model.config
    llm_cfg = cfg.llm_config
    if S_new is None:
        S_new = max(2, S_full // 3)
    B = 1  # <-- the whole point of this script
    C = cfg.num_audio_codebook
    kv_h = llm_cfg.num_key_value_heads
    d = llm_cfg.head_dim
    L = llm_cfg.num_hidden_layers

    torch.manual_seed(0)

    input_ids = torch.full((B, C, S_new), cfg.audio_mask_id, dtype=torch.long)
    # Half real text ids, half mask, on the single batch row.
    input_ids[0, :, : S_new // 2] = torch.randint(
        0, llm_cfg.vocab_size, (C, S_new // 2), dtype=torch.long
    )
    audio_mask = torch.ones((B, S_new), dtype=torch.bool)
    audio_mask[0, : S_new // 2] = False

    attention_mask = torch.ones((B, 1, S_new, S_full), dtype=torch.bool)

    offset0 = S_full - S_new
    tp_row0 = torch.arange(offset0, offset0 + S_new)
    target_positions = tp_row0.unsqueeze(0)  # (1, S_new)
    position_ids = target_positions.clone()

    past_k = [torch.randn(B, kv_h, S_full, d, dtype=torch.float32) for _ in range(L)]
    past_v = [torch.randn(B, kv_h, S_full, d, dtype=torch.float32) for _ in range(L)]

    return (
        input_ids, audio_mask, attention_mask, position_ids, target_positions,
        past_k, past_v,
    )


def _input_output_names(L: int):
    in_names = [
        "input_ids",
        "audio_mask",
        "attention_mask",
        "position_ids",
        "target_positions",
    ] + [f"past_key_{i}" for i in range(L)] + [f"past_value_{i}" for i in range(L)]

    out_names = (
        ["audio_logits"]
        + [f"present_key_{i}" for i in range(L)]
        + [f"present_value_{i}" for i in range(L)]
    )
    return in_names, out_names


def _dynamic_axes(L: int):
    # Same dynamic-axes map as the B=2 export. Even though we trace at B=1,
    # we keep the batch dim dynamic so the graph doesn't hard-code 1 into
    # every shape constraint — harmless, and future-proof if we ever want
    # to run multiple utterances in parallel on the same session.
    axes = {
        "input_ids": {0: "batch", 2: "seq_new"},
        "audio_mask": {0: "batch", 1: "seq_new"},
        "attention_mask": {0: "batch", 2: "seq_new", 3: "seq_full"},
        "position_ids": {0: "batch", 1: "seq_new"},
        "target_positions": {0: "batch", 1: "seq_new"},
        "audio_logits": {0: "batch", 2: "seq_new"},
    }
    for i in range(L):
        axes[f"past_key_{i}"] = {0: "batch", 2: "seq_full"}
        axes[f"past_value_{i}"] = {0: "batch", 2: "seq_full"}
        axes[f"present_key_{i}"] = {0: "batch", 2: "seq_full"}
        axes[f"present_value_{i}"] = {0: "batch", 2: "seq_full"}
    return axes


def main():
    from omnivoice.models.omnivoice import OmniVoice
    from kv_wrapper import OmniVoiceKvWrapper

    snap = resolve_hf_snapshot()
    print(f"[export-b1] snapshot: {snap}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _clear_output_files()

    t0 = time.time()
    model = OmniVoice.from_pretrained(
        str(snap), train=True, dtype=torch.float32, device_map="cpu"
    ).eval()
    model.llm.config._attn_implementation = "eager"
    print(f"[export-b1] loaded OmniVoice in {time.time() - t0:.1f}s")

    wrapper = OmniVoiceKvWrapper(model).eval()
    L = wrapper.num_layers

    S_full = 24
    S_new = 8
    (
        input_ids, audio_mask, attention_mask, position_ids, target_positions,
        past_k, past_v,
    ) = _example_inputs_b1(model, S_full=S_full, S_new=S_new)

    ref_out = wrapper(
        input_ids, audio_mask, attention_mask, position_ids, target_positions,
        *past_k, *past_v,
    )
    ref_logits = ref_out[0]
    print(f"[export-b1] wrapper dry-run ok. logits shape={tuple(ref_logits.shape)}")

    in_names, out_names = _input_output_names(L)
    dyn_axes = _dynamic_axes(L)

    print("[export-b1] tracing + serializing to ONNX...")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (
            input_ids, audio_mask, attention_mask, position_ids, target_positions,
            *past_k, *past_v,
        ),
        str(OUT_ONNX),
        input_names=in_names,
        output_names=out_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=dyn_axes,
        dynamo=False,
        export_params=True,
    )
    print(f"[export-b1] trace+save done in {time.time() - t0:.1f}s")

    print("[export-b1] converting to external data format...")
    t0 = time.time()
    m = onnx.load(str(OUT_ONNX), load_external_data=True)
    for init in m.graph.initializer:
        del init.external_data[:]
        init.data_location = onnx.TensorProto.DEFAULT

    _clear_output_files(keep={OUT_ONNX.name})

    onnx.save_model(
        m,
        str(OUT_ONNX),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=OUT_DATA_NAME,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"[export-b1] rewrote with external data in {time.time() - t0:.1f}s")

    graph_bytes = OUT_ONNX.stat().st_size
    data_bytes = (OUT_DIR / OUT_DATA_NAME).stat().st_size
    print(
        f"[export-b1] graph proto: {graph_bytes/1e6:.2f} MB   "
        f"external data: {data_bytes/1e9:.2f} GB"
    )

    import onnxruntime as ort

    print("[export-b1] loading with onnxruntime CPU and validating outputs...")
    t0 = time.time()
    sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
    print(f"[export-b1]   session created in {time.time() - t0:.1f}s")

    feed = {
        "input_ids": input_ids.numpy(),
        "audio_mask": audio_mask.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
        "target_positions": target_positions.numpy(),
    }
    for i in range(L):
        feed[f"past_key_{i}"] = past_k[i].numpy()
        feed[f"past_value_{i}"] = past_v[i].numpy()

    t0 = time.time()
    ort_out = sess.run(None, feed)
    print(f"[export-b1]   ORT run in {time.time() - t0:.1f}s")

    ort_logits = ort_out[0]
    diff = np.abs(ort_logits - ref_logits.numpy())
    print(
        f"[export-b1] ONNX vs PyTorch: max={diff.max():.4g}  mean={diff.mean():.4g}  "
        f"rel_mean={diff.mean() / max(1e-9, np.abs(ref_logits.numpy()).mean()) * 100:.3f}%"
    )
    if diff.max() > 1e-3:
        print("[export-b1] FAIL — ONNX graph does not match PyTorch wrapper to 1e-3")
        sys.exit(2)

    print("[export-b1] OK")


if __name__ == "__main__":
    main()
