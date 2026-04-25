/**
 * OmniVoice TTS Engine — headless inference via ONNX Runtime WebGPU.
 *
 * Adapted from VocoLoco's tts-worker.js for use in a headless Chromium browser
 * driven by Playwright. Runs OmniVoice iterative masked diffusion on WebGPU
 * (Vulkan backend) and returns PCM audio data.
 *
 * Key differences from VocoLoco:
 * - No Web Worker messaging — runs on main thread, exposes window.ttsEngine
 * - Models loaded from local HTTP server instead of CDN/cache
 * - Duration estimator inlined
 * - Returns PCM data directly instead of postMessage
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.0-dev.20260227-dce58e8711/dist/ort.all.mjs';
import { AutoTokenizer, env as tfEnv } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/dist/transformers.min.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.0-dev.20260227-dce58e8711/dist/';
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

// Tokenizer files live in the same Hugging Face repo as the ONNX bundle.
tfEnv.allowLocalModels = false;

let mainSession = null;
let decoderSession = null;
let encoderSession = null;
let gpuPostProc = null;
let asrModel = null;
let tokenizer = null;
let config = null;

// ─── KV-cached main model: architecture constants ───────────────────────────
// Coupled to the Qwen3 LLM used by OmniVoice (see omnivoice-kv-export/). The
// ONNX graph has 28 past_key_i/past_value_i inputs and the matching
// present_*_i outputs. kv_heads=8 and head_dim=128 drive the KV buffer
// shape: (batch=2, kv_heads, seq_full, head_dim) fp32 at the I/O boundary.
const KV_NUM_LAYERS = 28;
const KV_HEADS = 8;
const KV_HEAD_DIM = 128;
// Production main model. The bridge is intentionally standardized on this one
// artifact: fp16, KV-cached, B=1 split. Older monolithic/B=2/Q4/Q8 variants are
// export/benchmark artifacts only and are not probed at runtime.
const MAIN_MODEL_FILE = 'omnivoice-main-kv-fp16-b1.onnx';
const MAIN_MODEL_MANIFEST = 'omnivoice-main-kv-fp16-b1-manifest.json';
const MAIN_MODEL_VARIANT = 'kv-fp16-b1';
const DEFAULT_MODEL_REPO_ID = 'MarkShark2/omnivoice-onnx-kv-b1-fp16';
const DEFAULT_MODEL_BASE_URL = 'https://huggingface.co/MarkShark2/omnivoice-onnx-kv-b1-fp16/resolve/main';

// ─── ORT-WebGPU correctness diagnostics ────────────────────────────────────
// Keep all false for the normal fast path. These toggles isolate correctness
// regressions seen on newer ORT-WebGPU builds without changing model export.
const DIAG_FORCE_CPU_STAGED_POSTPROC = false; // download logits, then run WGSL post-proc
const DIAG_FORCE_JS_POSTPROC = false;         // download logits, then run JS post-proc
const DIAG_FORCE_WASM_DECODER = false;        // bypass WebGPU decoder
const DIAG_FORCE_CPU_KV_CACHE = false;        // keep present_kv on CPU between steps

let mainSessionIsKv = true;
let mainSessionIsKvB1 = true;
let mainSessionVariant = MAIN_MODEL_VARIANT;

// ─── Progress / status tracking ─────────────────────────────────────────────

window.ttsProgress = [];

function log(stage, detail) {
  const msg = `[${stage}] ${detail}`;
  console.log(msg);
  window.ttsProgress.push({ stage, detail, time: Date.now() });
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}

// ─── Tensor helper ──────────────────────────────────────────────────────────

function T(type, data, dims) { return new ort.Tensor(type, data, dims); }

// ─── Time steps (port of _get_time_steps) ───────────────────────────────────

function getTimeSteps(tStart, tEnd, numStep, tShift) {
  const steps = [];
  for (let i = 0; i <= numStep; i++) {
    let t = tStart + (tEnd - tStart) * (i / numStep);
    t = tShift * t / (1 + (tShift - 1) * t);
    steps.push(t);
  }
  return steps;
}

// ─── Seeded PRNG (mulberry32) ───────────────────────────────────────────────

function mulberry32(seed) {
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng = Math.random;

// ─── Duration estimation (inlined from duration-estimator.js) ───────────────

const WEIGHTS = {
  cjk: 3.0, hangul: 2.5, kana: 2.2, ethiopic: 3.0, yi: 3.0,
  indic: 1.8, thai_lao: 1.5, khmer_myanmar: 1.8,
  arabic: 1.5, hebrew: 1.5,
  latin: 1.0, cyrillic: 1.0, greek: 1.0, armenian: 1.0, georgian: 1.0,
  punctuation: 0.5, space: 0.2, digit: 3.5, mark: 0.0, default: 1.0,
};

const RANGES = [
  [0x02AF, 'latin'], [0x03FF, 'greek'], [0x052F, 'cyrillic'],
  [0x058F, 'armenian'], [0x05FF, 'hebrew'],
  [0x077F, 'arabic'], [0x089F, 'arabic'], [0x08FF, 'arabic'],
  [0x097F, 'indic'], [0x09FF, 'indic'], [0x0A7F, 'indic'],
  [0x0AFF, 'indic'], [0x0B7F, 'indic'], [0x0BFF, 'indic'],
  [0x0C7F, 'indic'], [0x0CFF, 'indic'], [0x0D7F, 'indic'],
  [0x0DFF, 'indic'], [0x0EFF, 'thai_lao'], [0x0FFF, 'indic'],
  [0x109F, 'khmer_myanmar'], [0x10FF, 'georgian'],
  [0x11FF, 'hangul'], [0x137F, 'ethiopic'], [0x139F, 'ethiopic'],
  [0x13FF, 'default'], [0x167F, 'default'], [0x169F, 'default'],
  [0x16FF, 'default'], [0x171F, 'default'], [0x173F, 'default'],
  [0x175F, 'default'], [0x177F, 'default'], [0x17FF, 'khmer_myanmar'],
  [0x18AF, 'default'], [0x18FF, 'default'],
  [0x194F, 'indic'], [0x19DF, 'indic'], [0x19FF, 'khmer_myanmar'],
  [0x1A1F, 'indic'], [0x1AAF, 'indic'], [0x1B7F, 'indic'],
  [0x1BBF, 'indic'], [0x1BFF, 'indic'], [0x1C4F, 'indic'],
  [0x1C7F, 'indic'], [0x1C8F, 'cyrillic'], [0x1CBF, 'georgian'],
  [0x1CCF, 'indic'], [0x1CFF, 'indic'], [0x1D7F, 'latin'],
  [0x1DBF, 'latin'], [0x1DFF, 'default'], [0x1EFF, 'latin'],
  [0x309F, 'kana'], [0x30FF, 'kana'], [0x312F, 'cjk'],
  [0x318F, 'hangul'], [0x9FFF, 'cjk'], [0xA4CF, 'yi'],
  [0xA4FF, 'default'], [0xA63F, 'default'], [0xA69F, 'cyrillic'],
  [0xA6FF, 'default'], [0xA7FF, 'latin'], [0xA82F, 'indic'],
  [0xA87F, 'default'], [0xA8DF, 'indic'], [0xA8FF, 'indic'],
  [0xA92F, 'indic'], [0xA95F, 'indic'], [0xA97F, 'hangul'],
  [0xA9DF, 'indic'], [0xA9FF, 'khmer_myanmar'], [0xAA5F, 'indic'],
  [0xAA7F, 'khmer_myanmar'], [0xAADF, 'indic'], [0xAAFF, 'indic'],
  [0xAB2F, 'ethiopic'], [0xAB6F, 'latin'], [0xABBF, 'default'],
  [0xABFF, 'indic'], [0xD7AF, 'hangul'], [0xFAFF, 'cjk'],
  [0xFDFF, 'arabic'], [0xFE6F, 'default'], [0xFEFF, 'arabic'],
  [0xFFEF, 'latin'],
];

const BREAKPOINTS = RANGES.map(r => r[0]);

function bisectLeft(arr, val) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < val) lo = mid + 1; else hi = mid;
  }
  return lo;
}

function getCharCategory(code) {
  if ((code >= 0x0300 && code <= 0x036F) ||
    (code >= 0x0483 && code <= 0x0489) ||
    (code >= 0x0591 && code <= 0x05BD) ||
    (code >= 0x064B && code <= 0x065F) ||
    (code >= 0x0900 && code <= 0x0903) ||
    (code >= 0x093A && code <= 0x094F) ||
    (code >= 0x0951 && code <= 0x0957) ||
    (code >= 0x0962 && code <= 0x0963) ||
    (code >= 0xFE20 && code <= 0xFE2F)) return 'M';
  if ((code >= 0x0021 && code <= 0x002F) ||
    (code >= 0x003A && code <= 0x0040) ||
    (code >= 0x005B && code <= 0x0060) ||
    (code >= 0x007B && code <= 0x007E) ||
    (code >= 0x2000 && code <= 0x206F) ||
    (code >= 0x3000 && code <= 0x303F)) return 'P';
  if (code >= 0x0030 && code <= 0x0039) return 'N';
  if ((code >= 0x00A0 && code <= 0x00BF) ||
    (code >= 0x2100 && code <= 0x27FF)) return 'S';
  if (code === 0x00A0 || code === 0x2000 || code === 0x2001 ||
    code === 0x2002 || code === 0x2003 || code === 0x3000) return 'Z';
  return 'L';
}

function getCharWeight(char) {
  const code = char.codePointAt(0);
  if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return WEIGHTS.latin;
  if (code === 32) return WEIGHTS.space;
  if (code === 0x0640) return WEIGHTS.mark;
  const cat = getCharCategory(code);
  if (cat === 'M') return WEIGHTS.mark;
  if (cat === 'P' || cat === 'S') return WEIGHTS.punctuation;
  if (cat === 'Z') return WEIGHTS.space;
  if (cat === 'N') return WEIGHTS.digit;
  const idx = bisectLeft(BREAKPOINTS, code);
  if (idx < RANGES.length) return WEIGHTS[RANGES[idx][1]] || WEIGHTS.default;
  if (code > 0x20000) return WEIGHTS.cjk;
  return WEIGHTS.default;
}

function calculateTotalWeight(text) {
  let total = 0;
  for (const char of text) total += getCharWeight(char);
  return total;
}

function estimateDuration(targetText, refText, refDuration, lowThreshold = 50, boostStrength = 3) {
  if (refDuration <= 0 || !refText) return 0;
  const refWeight = calculateTotalWeight(refText);
  if (refWeight === 0) return 0;
  const speedFactor = refWeight / refDuration;
  const targetWeight = calculateTotalWeight(targetText);
  const estimated = targetWeight / speedFactor;
  if (lowThreshold !== null && estimated < lowThreshold) {
    const alpha = 1.0 / boostStrength;
    return lowThreshold * Math.pow(estimated / lowThreshold, alpha);
  }
  return estimated;
}

function estimateTargetTokens(text, refText = null, numRefAudioTokens = null, speed = 1.0) {
  let rText = refText;
  let rTokens = numRefAudioTokens;
  if (rTokens === null || rText === null || !rText.length) {
    rText = 'Nice to meet you.';
    rTokens = 25;
  }
  let est = estimateDuration(text, rText, rTokens);
  if (speed > 0 && speed !== 1.0) est = est / speed;
  return Math.max(1, Math.round(est));
}

// ─── Log-softmax + CFG post-processing (CPU path) ──────────────────────────

const _cLP = new Float32Array(1025);
const _uLP = new Float32Array(1025);
const _g = new Float32Array(1025);

function logSoftmaxInto(arr, offset, len, out) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) { const v = arr[offset + i]; if (v > max) max = v; }
  let sum = 0;
  for (let i = 0; i < len; i++) sum += Math.exp(arr[offset + i] - max);
  const lse = max + Math.log(sum);
  for (let i = 0; i < len; i++) out[i] = arr[offset + i] - lse;
}

function normalizeFp16LogitsToF32(logits) {
  if (typeof Float16Array !== 'undefined' && logits instanceof Float16Array) {
    return new Float32Array(logits);
  }
  if (logits instanceof Uint16Array) {
    return new Float32Array(new Float16Array(logits.buffer, logits.byteOffset, logits.length));
  }
  return logits;
}

function cpuPostProcess(logits, C, maxLen, V, numTargetTokens, targetOff, maskId, guidanceScale, layerPenalty, pred, scores) {
  // audio_logits is fp16 in the current model (see optimization #7 in README).
  // ORT-web returns Float16Array on modern builds, Uint16Array (raw fp16 bits)
  // on older ones. Normalize to Float32Array once; the per-V inner loops stay
  // hot-path simple. Float32Array passes through unchanged for older graphs.
  logits = normalizeFp16LogitsToF32(logits);
  const gScale1 = 1 + guidanceScale;
  for (let c = 0; c < C; c++) {
    const layerScore = layerPenalty * c;
    for (let t = 0; t < numTargetTokens; t++) {
      const cOff = (c * maxLen + targetOff + t) * V;
      const uOff = ((C + c) * maxLen + t) * V;
      logSoftmaxInto(logits, cOff, V, _cLP);
      logSoftmaxInto(logits, uOff, V, _uLP);
      let mx = -Infinity;
      for (let v = 0; v < V; v++) {
        const gv = gScale1 * _cLP[v] - guidanceScale * _uLP[v];
        _g[v] = gv;
        if (gv > mx) mx = gv;
      }
      let sm = 0;
      for (let v = 0; v < V; v++) sm += Math.exp(_g[v] - mx);
      const lse = mx + Math.log(sm);
      let bestV = 0, bestS = -Infinity;
      for (let v = 0; v < V; v++) {
        if (v === maskId) continue;
        const lp = _g[v] - lse;
        if (lp > bestS) { bestS = lp; bestV = v; }
      }
      const idx = c * numTargetTokens + t;
      pred[idx] = bestV;
      scores[idx] = bestS - layerScore;
    }
  }
}

function cpuPostProcessSplit(condLogits, uncondLogits, C, condMaxLen, condTargetOff, uncondMaxLen, uncondTargetOff, V, numTargetTokens, maskId, guidanceScale, layerPenalty, pred, scores) {
  condLogits = normalizeFp16LogitsToF32(condLogits);
  uncondLogits = normalizeFp16LogitsToF32(uncondLogits);
  const gScale1 = 1 + guidanceScale;
  for (let c = 0; c < C; c++) {
    const layerScore = layerPenalty * c;
    for (let t = 0; t < numTargetTokens; t++) {
      const cOff = (c * condMaxLen + condTargetOff + t) * V;
      const uOff = (c * uncondMaxLen + uncondTargetOff + t) * V;
      logSoftmaxInto(condLogits, cOff, V, _cLP);
      logSoftmaxInto(uncondLogits, uOff, V, _uLP);
      let mx = -Infinity;
      for (let v = 0; v < V; v++) {
        const gv = gScale1 * _cLP[v] - guidanceScale * _uLP[v];
        _g[v] = gv;
        if (gv > mx) mx = gv;
      }
      let sm = 0;
      for (let v = 0; v < V; v++) sm += Math.exp(_g[v] - mx);
      const lse = mx + Math.log(sm);
      let bestV = 0, bestS = -Infinity;
      for (let v = 0; v < V; v++) {
        if (v === maskId) continue;
        const lp = _g[v] - lse;
        if (lp > bestS) { bestS = lp; bestV = v; }
      }
      const idx = c * numTargetTokens + t;
      pred[idx] = bestV;
      scores[idx] = bestS - layerScore;
    }
  }
}

// ─── Prepare inference inputs ───────────────────────────────────────────────

async function prepareInferenceInputs(text, numTargetTokens, tok, cfg, opts = {}) {
  const { refText = null, refAudioTokens = null, lang = null, instruct = null, denoise = true } = opts;
  const C = cfg.num_audio_codebook;
  const maskId = cfg.audio_mask_id;

  // Build style string
  let styleText = '';
  if (denoise) styleText += '<|denoise|>';
  styleText += `<|lang_start|>${lang || 'None'}<|lang_end|>`;
  styleText += `<|instruct_start|>${instruct || 'None'}<|instruct_end|>`;

  // Build text string
  let fullText = refText ? refText.trim() + ' ' + text.trim() : text.trim();
  fullText = fullText.replace(/[\r\n]+/g, '').replace(/[ \t]+/g, ' ');
  const wrappedText = `<|text_start|>${fullText}<|text_end|>`;

  // Tokenize using transformers.js (Qwen2 BPE)
  const styleEncoded = await tok(styleText, { add_special_tokens: false });
  const textEncoded = await tok(wrappedText, { add_special_tokens: false });
  const styleIds = Array.from(styleEncoded.input_ids.data, Number);
  const textIds = Array.from(textEncoded.input_ids.data, Number);

  // Sequence layout: [style | text | ref_audio? | target_masked]
  const refLen = refAudioTokens ? refAudioTokens[0].length : 0;
  const totalLen = styleIds.length + textIds.length + refLen + numTargetTokens;

  const inputIds = new BigInt64Array(C * totalLen);

  // Style tokens (replicated across codebooks)
  for (let c = 0; c < C; c++)
    for (let i = 0; i < styleIds.length; i++)
      inputIds[c * totalLen + i] = BigInt(styleIds[i]);

  // Text tokens
  const textOff = styleIds.length;
  for (let c = 0; c < C; c++)
    for (let i = 0; i < textIds.length; i++)
      inputIds[c * totalLen + textOff + i] = BigInt(textIds[i]);

  // Reference audio tokens
  const refOff = textOff + textIds.length;
  if (refAudioTokens) {
    for (let c = 0; c < C; c++)
      for (let t = 0; t < refLen; t++)
        inputIds[c * totalLen + refOff + t] = BigInt(refAudioTokens[c][t]);
  }

  // Target = all mask
  const targetOff = refOff + refLen;
  for (let c = 0; c < C; c++)
    for (let t = 0; t < numTargetTokens; t++)
      inputIds[c * totalLen + targetOff + t] = BigInt(maskId);

  // Audio mask: true for audio positions (ref + target)
  const audioMask = new Uint8Array(totalLen);
  const audioStart = refAudioTokens ? refOff : targetOff;
  for (let i = audioStart; i < totalLen; i++) audioMask[i] = 1;

  return { inputIds, audioMask, totalLen, numTargetTokens, targetOff, C };
}

async function ensureAsrModel() {
  if (!asrModel || typeof asrModel !== 'function') {
    log('downloading', 'Lazy-loading whisper-base.en (~140MB)...');
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
    env.allowLocalModels = false;
    asrModel = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base.en', {
      device: 'webgpu',
    });
    log('init', 'Whisper ASR loaded dynamically.');
  }
  return asrModel;
}

function unloadAsrModel() {
  if (asrModel) {
    asrModel = null;
    if (globalThis.gc) {
      globalThis.gc();
    }
    log('encoding', 'Whisper ASR unloaded from memory.');
  }
}

// ─── Top-k unmask ───────────────────────────────────────────────────────────

let topKIndices = null;

function swapInt32(arr, i, j) {
  const tmp = arr[i];
  arr[i] = arr[j];
  arr[j] = tmp;
}

function partitionTopK(indices, left, right, pivotIndex, scores) {
  const pivotScore = scores[indices[pivotIndex]];
  swapInt32(indices, pivotIndex, right);
  let store = left;
  for (let i = left; i < right; i++) {
    if (scores[indices[i]] > pivotScore) {
      swapInt32(indices, store, i);
      store++;
    }
  }
  swapInt32(indices, right, store);
  return store;
}

function quickselectTopK(indices, count, k, scores) {
  let left = 0;
  let right = count - 1;
  const target = k - 1;
  while (left < right) {
    const pivotIndex = left + ((right - left) >> 1);
    const pivotNewIndex = partitionTopK(indices, left, right, pivotIndex, scores);
    if (pivotNewIndex === target) return;
    if (pivotNewIndex > target) right = pivotNewIndex - 1;
    else left = pivotNewIndex + 1;
  }
}

function topKUnmask(scores, pred, tokens, n, k) {
  if (!topKIndices || topKIndices.length < n) topKIndices = new Int32Array(n);
  const indices = topKIndices;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (scores[i] > -Infinity) indices[count++] = i;
  }
  const limit = Math.min(k, count);
  if (limit <= 0) return;
  if (limit < count) quickselectTopK(indices, count, limit, scores);
  for (let i = 0; i < limit; i++) {
    tokens[indices[i]] = BigInt(pred[indices[i]]);
  }
}

let pred_buf = null, scores_buf = null;

// ─── Iterative unmasking generation loop ────────────────────────────────────

async function generateIterative(inp, cfg, numStep, guidanceScale, tShift, layerPenalty = 5.0, posTemp = 5.0) {
  const { inputIds, audioMask, totalLen, numTargetTokens, targetOff, C } = inp;
  const maskId = cfg.audio_mask_id;
  const V = cfg.audio_vocab_size;

  const condLen = totalLen;
  const uncondLen = numTargetTokens;
  const maxLen = condLen;

  // ── Static batch buffers shared by both the monolithic path and the
  //    KV path's step-0 (prefix) call. Shape/content identical to what the
  //    monolithic graph has always taken: (cond row, uncond row) in a batch.
  //    In KV mode we additionally carry position_ids + target_positions.

  // Batch input_ids: (2, C, maxLen) — cond + uncond
  const bIds = new BigInt64Array(2 * C * maxLen).fill(BigInt(maskId));
  for (let c = 0; c < C; c++)
    for (let s = 0; s < condLen; s++)
      bIds[c * maxLen + s] = inputIds[c * totalLen + s];
  for (let c = 0; c < C; c++)
    for (let t = 0; t < uncondLen; t++)
      bIds[(C + c) * maxLen + t] = inputIds[c * totalLen + targetOff + t];

  // Batch audio_mask: (2, maxLen)
  const bMask = new Uint8Array(2 * maxLen);
  for (let s = 0; s < condLen; s++) bMask[s] = audioMask[s];
  for (let t = 0; t < uncondLen; t++) bMask[maxLen + t] = 1;

  // Batch attention_mask: (2, 1, maxLen, maxLen)
  const bAttn = new Uint8Array(2 * maxLen * maxLen);
  for (let q = 0; q < condLen; q++)
    for (let k = 0; k < condLen; k++)
      bAttn[q * maxLen + k] = 1;
  for (let q = 0; q < uncondLen; q++)
    for (let k = 0; k < uncondLen; k++)
      bAttn[maxLen * maxLen + q * maxLen + k] = 1;
  for (let p = uncondLen; p < maxLen; p++)
    bAttn[maxLen * maxLen + p * maxLen + p] = 1;

  // Token state
  const tokens = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
  pred_buf = null; scores_buf = null;

  // Unmasking schedule
  const timesteps = getTimeSteps(0, 1, numStep + 1, tShift);
  const totalMask = numTargetTokens * C;
  let rem = totalMask;
  const sched = [];
  for (let s = 0; s < numStep; s++) {
    const n = s === numStep - 1 ? rem : Math.min(Math.ceil(totalMask * (timesteps[s + 1] - timesteps[s])), rem);
    sched.push(n); rem -= n;
  }

  if (gpuPostProc) {
    try { gpuPostProc.prepare(C, maxLen, V, numTargetTokens); }
    catch (e) { log('error', `GPU prepare failed: ${e.message}`); gpuPostProc.destroy(); gpuPostProc = null; }
  }

  // ── KV-mode-only pre-allocated state ────────────────────────────────────
  // Prefix position/target_positions (step 0): arange broadcast across batch.
  // Step-mode buffers (step 1..N-1): target-only inputs + wider attention mask
  //   that attends to the full cached prefix. Only input_ids changes per step.
  //
  // In B=1 split mode (mainSessionIsKvB1), every per-step tensor exists in
  // TWO independent (B=1) flavours — cond and uncond. The cond call sees the
  // full prefix (S_full = maxLen); the uncond call only ever sees its own
  // target-only window (S_full = uncondLen = numTargetTokens), which is the
  // whole point of the split. Frozen prefix K/V is likewise stored in two
  // independent arrays.
  let prefixPosIds = null;
  let stepBIds = null, stepBMask = null, stepBAttn = null;
  let stepPosIds = null, stepTargetPos = null;
  let prefixPastKv = null; // B=2 path only: zeros fed in for step-0 prefix call

  // B=1 split state. All shapes start with a leading B=1 dim.
  let b1CondIds = null, b1CondMask = null, b1CondAttn = null, b1CondPos = null;
  let b1UncondIds = null, b1UncondMask = null, b1UncondAttn = null, b1UncondPos = null;
  let b1StepCondIds = null, b1StepCondAttn = null, b1StepCondPos = null, b1StepCondMask = null;
  let b1StepUncondIds = null, b1StepUncondAttn = null, b1StepUncondPos = null, b1StepUncondMask = null;
  let b1PrefixCondZeros = null, b1PrefixUncondZeros = null;

  // Step-0 presents are kept alive for the rest of the loop (frozen prefix K/V).
  // In B=2: single array of 2*KV_NUM_LAYERS tensors, each (B=2, kv_h, maxLen, d).
  // In B=1: two independent arrays, cond (S_full=maxLen) + uncond (S_full=uncondLen).
  let frozenPresentKv = null;
  let frozenPresentKvCond = null, frozenPresentKvUncond = null;
  let uncondRunsExecuted = 0;  // for the chunk-end perf log

  if (mainSessionIsKv && !mainSessionIsKvB1) {
    // ── B=2 KV path (legacy, optimization 5) ───────────────────────────────
    prefixPosIds = new BigInt64Array(2 * maxLen);
    for (let s = 0; s < maxLen; s++) {
      prefixPosIds[s] = BigInt(s);
      prefixPosIds[maxLen + s] = BigInt(s);
    }

    // Step-mode inputs: target-only (numTargetTokens along seq_new), S_full=maxLen.
    stepBIds = new BigInt64Array(2 * C * numTargetTokens).fill(BigInt(maskId));
    stepBMask = new Uint8Array(2 * numTargetTokens).fill(1);
    // attention (2, 1, numTargetTokens, maxLen):
    //   cond (b=0):  target queries attend to ALL keys   → ones
    //   uncond (b=1):target queries attend only to keys < uncondLen (= numTargetTokens)
    stepBAttn = new Uint8Array(2 * numTargetTokens * maxLen);
    for (let q = 0; q < numTargetTokens; q++)
      for (let k = 0; k < maxLen; k++)
        stepBAttn[q * maxLen + k] = 1;
    for (let q = 0; q < numTargetTokens; q++)
      for (let k = 0; k < uncondLen; k++)
        stepBAttn[numTargetTokens * maxLen + q * maxLen + k] = 1;

    stepPosIds = new BigInt64Array(2 * numTargetTokens);
    for (let t = 0; t < numTargetTokens; t++) {
      stepPosIds[t] = BigInt(targetOff + t);
      stepPosIds[numTargetTokens + t] = BigInt(t);
    }
    stepTargetPos = new BigInt64Array(stepPosIds);

    const kvElems = 2 * KV_HEADS * maxLen * KV_HEAD_DIM;
    prefixPastKv = { zeros: new Float16Array(kvElems), shape: [2, KV_HEADS, maxLen, KV_HEAD_DIM] };
    frozenPresentKv = null;
  } else if (mainSessionIsKvB1) {
    // ── B=1 KV split path (optimization 8) ─────────────────────────────────
    // Build per-side prefix (step-0) and step (step>=1) input buffers. Each
    // one starts with a leading B=1 dim.

    // Step-0 cond inputs: the full (text + ref + target-masked) sequence.
    // Reuses bIds's cond row (already filled above) — avoid doubling memory.
    b1CondIds = new BigInt64Array(C * maxLen);
    for (let c = 0; c < C; c++)
      for (let s = 0; s < maxLen; s++)
        b1CondIds[c * maxLen + s] = bIds[c * maxLen + s];
    b1CondMask = new Uint8Array(maxLen);
    for (let s = 0; s < maxLen; s++) b1CondMask[s] = bMask[s];
    b1CondAttn = new Uint8Array(maxLen * maxLen);
    for (let q = 0; q < maxLen; q++)
      for (let k = 0; k < maxLen; k++)
        b1CondAttn[q * maxLen + k] = 1;
    b1CondPos = new BigInt64Array(maxLen);
    for (let s = 0; s < maxLen; s++) b1CondPos[s] = BigInt(s);

    // Step-0 uncond inputs: target-only sequence (S = uncondLen == numTargetTokens).
    // All audio mask tokens, full attention.
    b1UncondIds = new BigInt64Array(C * uncondLen).fill(BigInt(maskId));
    b1UncondMask = new Uint8Array(uncondLen).fill(1);
    b1UncondAttn = new Uint8Array(uncondLen * uncondLen);
    for (let q = 0; q < uncondLen; q++)
      for (let k = 0; k < uncondLen; k++)
        b1UncondAttn[q * uncondLen + k] = 1;
    b1UncondPos = new BigInt64Array(uncondLen);
    for (let s = 0; s < uncondLen; s++) b1UncondPos[s] = BigInt(s);

    // Step>=1 cond inputs: S_new = numTargetTokens, S_full = maxLen.
    b1StepCondIds = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
    b1StepCondMask = new Uint8Array(numTargetTokens).fill(1);
    b1StepCondAttn = new Uint8Array(numTargetTokens * maxLen);
    for (let q = 0; q < numTargetTokens; q++)
      for (let k = 0; k < maxLen; k++)
        b1StepCondAttn[q * maxLen + k] = 1;
    b1StepCondPos = new BigInt64Array(numTargetTokens);
    for (let t = 0; t < numTargetTokens; t++) b1StepCondPos[t] = BigInt(targetOff + t);

    // Step>=1 uncond inputs: S_new = numTargetTokens, S_full = uncondLen.
    // Since uncondLen == numTargetTokens, the step-mode query and key counts
    // are identical for uncond. The attention matrix is num × num.
    b1StepUncondIds = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
    b1StepUncondMask = new Uint8Array(numTargetTokens).fill(1);
    b1StepUncondAttn = new Uint8Array(numTargetTokens * uncondLen);
    for (let q = 0; q < numTargetTokens; q++)
      for (let k = 0; k < uncondLen; k++)
        b1StepUncondAttn[q * uncondLen + k] = 1;
    b1StepUncondPos = new BigInt64Array(numTargetTokens);
    for (let t = 0; t < numTargetTokens; t++) b1StepUncondPos[t] = BigInt(t);

    // Zero past_kv buffers for step-0 call. Cond needs S_full=maxLen,
    // uncond needs S_full=uncondLen. Each buffer is reused across all 56
    // past_{key,value}_i inputs on its respective side.
    b1PrefixCondZeros = {
      zeros: new Float16Array(KV_HEADS * maxLen * KV_HEAD_DIM),
      shape: [1, KV_HEADS, maxLen, KV_HEAD_DIM],
    };
    b1PrefixUncondZeros = {
      zeros: new Float16Array(KV_HEADS * uncondLen * KV_HEAD_DIM),
      shape: [1, KV_HEADS, uncondLen, KV_HEAD_DIM],
    };

    frozenPresentKvCond = null;
    frozenPresentKvUncond = null;
  }

  let totalInferenceMs = 0, totalModelMs = 0, totalPostProcMs = 0, totalSampleMs = 0;
  window._abortRequested = false;
  for (let step = 0; step < numStep; step++) {
    if (window._abortRequested) {
      log('generating', 'Generation aborted by client signal.');
      throw new Error('ABORTED_BY_CLIENT');
    }
    const k = sched[step];
    if (k <= 0) continue;
    const stepT0 = performance.now();

    const modelT0 = performance.now();

    // ── Build per-step input tensors ──────────────────────────────────────
    // Four cases now:
    //   A) monolithic: (input_ids, audio_mask, attention_mask) single B=2 run
    //   B) KV B=2 prefix (step 0 when mainSessionIsKv && !mainSessionIsKvB1):
    //      B=2 inputs + position_ids + target_positions + 56 past_kv zeros
    //   C) KV B=2 step (step 1..N-1): target-only B=2 inputs + 56 past_kv from
    //      frozen-prefix GPU buffers saved at step 0
    //   D) KV B=1 split (mainSessionIsKvB1): TWO separate B=1 mainSession.run
    //      calls per step, one cond, one uncond. Each side has its own
    //      frozen-prefix KV snapshot and its own logits output buffer.

    const nPos = C * numTargetTokens;
    const pred = step === 0 ? new Int32Array(nPos) : pred_buf;
    const scores = step === 0 ? new Float32Array(nPos) : scores_buf;
    if (step === 0) { pred_buf = pred; scores_buf = scores; }

    if (mainSessionIsKvB1) {
      // ── Case D: B=1 split ─────────────────────────────────────────────
      let condTensors;
      let uncondTensors;
      let condLogitsSeqLen, condLogitsTargetOff, uncondLogitsSeqLen;

      if (step === 0) {
        // Cond prefix: (1, C, maxLen) inputs, S_full = maxLen, pos = [0..maxLen).
        condTensors = {
          input_ids: T('int64', b1CondIds, [1, C, maxLen]),
          audio_mask: T('bool', b1CondMask, [1, maxLen]),
          attention_mask: T('bool', b1CondAttn, [1, 1, maxLen, maxLen]),
          position_ids: T('int64', b1CondPos, [1, maxLen]),
          target_positions: T('int64', b1CondPos.slice(), [1, maxLen]),
        };
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          condTensors[`past_key_${i}`] = T('float16', b1PrefixCondZeros.zeros, b1PrefixCondZeros.shape);
          condTensors[`past_value_${i}`] = T('float16', b1PrefixCondZeros.zeros, b1PrefixCondZeros.shape);
        }
        // Uncond prefix: (1, C, uncondLen) inputs, S_full = uncondLen.
        uncondTensors = {
          input_ids: T('int64', b1UncondIds, [1, C, uncondLen]),
          audio_mask: T('bool', b1UncondMask, [1, uncondLen]),
          attention_mask: T('bool', b1UncondAttn, [1, 1, uncondLen, uncondLen]),
          position_ids: T('int64', b1UncondPos, [1, uncondLen]),
          target_positions: T('int64', b1UncondPos.slice(), [1, uncondLen]),
        };
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          uncondTensors[`past_key_${i}`] = T('float16', b1PrefixUncondZeros.zeros, b1PrefixUncondZeros.shape);
          uncondTensors[`past_value_${i}`] = T('float16', b1PrefixUncondZeros.zeros, b1PrefixUncondZeros.shape);
        }
        condLogitsSeqLen = maxLen;   condLogitsTargetOff = targetOff;
        uncondLogitsSeqLen = uncondLen;
      } else {
        // Target-mode: rebuild cond input_ids from the monolithic bIds
        // buffer (tokens get written back to bIds at the end of every step
        // so the builder stays simple here). Both branches run every step.
        for (let c = 0; c < C; c++) {
          for (let t = 0; t < numTargetTokens; t++) {
            const v = bIds[c * maxLen + targetOff + t];
            b1StepCondIds[c * numTargetTokens + t] = v;
            b1StepUncondIds[c * numTargetTokens + t] = v;
          }
        }
        condTensors = {
          input_ids: T('int64', b1StepCondIds, [1, C, numTargetTokens]),
          audio_mask: T('bool', b1StepCondMask, [1, numTargetTokens]),
          attention_mask: T('bool', b1StepCondAttn, [1, 1, numTargetTokens, maxLen]),
          position_ids: T('int64', b1StepCondPos, [1, numTargetTokens]),
          target_positions: T('int64', b1StepCondPos.slice(), [1, numTargetTokens]),
        };
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          condTensors[`past_key_${i}`] = frozenPresentKvCond[i];
          condTensors[`past_value_${i}`] = frozenPresentKvCond[i + KV_NUM_LAYERS];
        }
        uncondTensors = {
          input_ids: T('int64', b1StepUncondIds, [1, C, numTargetTokens]),
          audio_mask: T('bool', b1StepUncondMask, [1, numTargetTokens]),
          attention_mask: T('bool', b1StepUncondAttn, [1, 1, numTargetTokens, uncondLen]),
          position_ids: T('int64', b1StepUncondPos, [1, numTargetTokens]),
          target_positions: T('int64', b1StepUncondPos.slice(), [1, numTargetTokens]),
        };
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          uncondTensors[`past_key_${i}`] = frozenPresentKvUncond[i];
          uncondTensors[`past_value_${i}`] = frozenPresentKvUncond[i + KV_NUM_LAYERS];
        }
        condLogitsSeqLen = numTargetTokens;   condLogitsTargetOff = 0;
        uncondLogitsSeqLen = numTargetTokens;
      }

      // Sequential cond then uncond. ORT-web's InferenceSession
      // is not reentrant — calling run() while another run() is still
      // pending on the same session throws "Session already started" from
      // the IOBinding path. The only way to actually parallelize would be
      // to load the main model into a second sibling session, which
      // roughly doubles VRAM for marginal wall-time gain (WebGPU's queue
      // still serializes). Not worth it at our current perf ceiling.
      const condResults = await mainSession.run(condTensors);
      const uncondResults = await mainSession.run(uncondTensors);
      uncondRunsExecuted++;

      const condLogitsTensor = condResults.audio_logits;
      const uncondLogitsTensor = uncondResults.audio_logits;
      const canZeroCopy =
        !DIAG_FORCE_CPU_STAGED_POSTPROC &&
        !DIAG_FORCE_JS_POSTPROC &&
        gpuPostProc &&
        gpuPostProc.sharedWithOrt &&
        condLogitsTensor.location === 'gpu-buffer' &&
        uncondLogitsTensor.location === 'gpu-buffer' &&
        condLogitsTensor.gpuBuffer &&
        uncondLogitsTensor.gpuBuffer;

      // Dispose per-step input tensors we own. Past K/V at step>=1 are the
      // frozen prefix buffers — don't dispose them here.
      for (const name of Object.keys(condTensors)) {
        if (step > 0 && (name.startsWith('past_key_') || name.startsWith('past_value_'))) continue;
        condTensors[name].dispose();
      }
      for (const name of Object.keys(uncondTensors)) {
        if (step > 0 && (name.startsWith('past_key_') || name.startsWith('past_value_'))) continue;
        uncondTensors[name].dispose();
      }

      // Save (or discard) the two sides' present_kv.
      if (step === 0) {
        frozenPresentKvCond = new Array(2 * KV_NUM_LAYERS);
        frozenPresentKvUncond = new Array(2 * KV_NUM_LAYERS);
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          frozenPresentKvCond[i]                    = condResults[`present_key_${i}`];
          frozenPresentKvCond[i + KV_NUM_LAYERS]    = condResults[`present_value_${i}`];
          frozenPresentKvUncond[i]                  = uncondResults[`present_key_${i}`];
          frozenPresentKvUncond[i + KV_NUM_LAYERS]  = uncondResults[`present_value_${i}`];
        }
        const nGpuC = frozenPresentKvCond.filter(t => t.location === 'gpu-buffer').length;
        const nGpuU = frozenPresentKvUncond.filter(t => t.location === 'gpu-buffer').length;
        log('generating', `KV-B1 diag: present gpu-buffer cond=${nGpuC}/${frozenPresentKvCond.length}, uncond=${nGpuU}/${frozenPresentKvUncond.length}, logits cond=${condLogitsTensor.location} uncond=${uncondLogitsTensor.location}, S_full_c=${maxLen} S_full_u=${uncondLen} S_new=${numTargetTokens}`);
      } else {
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          condResults[`present_key_${i}`].dispose();
          condResults[`present_value_${i}`].dispose();
          uncondResults[`present_key_${i}`].dispose();
          uncondResults[`present_value_${i}`].dispose();
        }
      }

      totalModelMs += performance.now() - modelT0;

      // Post-proc: split path sets uncondBatchOff=0, per-side seq_len/off.
      const ppParams = {
        C, V, numTargetTokens, maskId, guidanceScale, layerPenalty,
        // Legacy aliases for helpers that still read these (used only if the
        // split path falls back to the B=2-shaped run()).
        maxLen: condLogitsSeqLen, targetOff: condLogitsTargetOff,
      };
      const ppExt = {
        condMaxLen: condLogitsSeqLen, condTargetOff: condLogitsTargetOff,
        uncondMaxLen: uncondLogitsSeqLen, uncondTargetOff: 0,
      };

      const postProcT0 = performance.now();
      let usedGpu = false;
      if (canZeroCopy) {
        try {
          // B=1 logits: (1, C, seq_len, V) fp16 → 2 bytes/elem, no batch prefix.
          // (The 2 bytes at the end is bytes-per-fp16-element; the previous
          // B=2 formula's leading `2 *` was the batch size, not a dtype factor.)
          const condBytes   = C * condLogitsSeqLen   * V * 2;
          const uncondBytes = C * uncondLogitsSeqLen * V * 2;
          await gpuPostProc.runGpuSplit(
            condLogitsTensor.gpuBuffer, uncondLogitsTensor.gpuBuffer,
            condBytes, uncondBytes,
            ppParams, ppExt,
            pred, scores,
          );
          usedGpu = true;
        } catch (e) {
          log('warning', `B=1 split zero-copy post-proc failed (${e.message}); disabling zero-copy and falling back.`);
          gpuPostProc.sharedWithOrt = false;
        }
      }
      if (!usedGpu && DIAG_FORCE_JS_POSTPROC) {
        const condData = typeof condLogitsTensor.getData === 'function'
          ? await condLogitsTensor.getData() : condLogitsTensor.data.slice();
        const uncondData = typeof uncondLogitsTensor.getData === 'function'
          ? await uncondLogitsTensor.getData() : uncondLogitsTensor.data.slice();
        cpuPostProcessSplit(
          condData, uncondData,
          C, condLogitsSeqLen, condLogitsTargetOff,
          uncondLogitsSeqLen, 0,
          V, numTargetTokens, maskId, guidanceScale, layerPenalty,
          pred, scores,
        );
      } else if (!usedGpu) {
        // CPU-staged fallback for the split path: stage both buffers and run
        // the same shader. The `run()` helper only supports the B=2 single-
        // buffer layout, so we splice the two fp16 arrays into one big
        // (2, C, maxLen', V) buffer shaped like the B=2 logits tensor.
        const condData = typeof condLogitsTensor.getData === 'function'
          ? await condLogitsTensor.getData() : condLogitsTensor.data.slice();
        const uncondData = typeof uncondLogitsTensor.getData === 'function'
          ? await uncondLogitsTensor.getData() : uncondLogitsTensor.data.slice();
        // Build a synthetic (2, C, condLogitsSeqLen, V) buffer where the
        // uncond half is padded with -inf fp16 outside [0..uncondLogitsSeqLen).
        // Easier: two separate cpuPostProcess passes would require reworking
        // cpuPostProcess to take two buffers; for now, synthesize a combined
        // layout that matches the B=2 shader addressing with condMaxLen ==
        // uncondMaxLen == condLogitsSeqLen. We zero-pad uncond; since the
        // CFG fusion only reads positions [0..numTargetTokens) it never
        // touches the padding.
        const condElems = C * condLogitsSeqLen * V;
        const uncondElems = C * condLogitsSeqLen * V;
        const combined = new Uint16Array(condElems + uncondElems);
        combined.set(new Uint16Array(condData.buffer, condData.byteOffset, condElems), 0);
        // Copy uncond row-by-row (its row stride is uncondLogitsSeqLen*V, but
        // the combined buffer allocates condLogitsSeqLen*V per row).
        const uncondSrc = new Uint16Array(uncondData.buffer, uncondData.byteOffset, C * uncondLogitsSeqLen * V);
        for (let c = 0; c < C; c++) {
          combined.set(
            uncondSrc.subarray(c * uncondLogitsSeqLen * V, (c + 1) * uncondLogitsSeqLen * V),
            condElems + c * condLogitsSeqLen * V,
          );
        }
        await gpuPostProc.run(combined, ppParams, pred, scores);
      }
      totalPostProcMs += performance.now() - postProcT0;

      if (step === 0) {
        let nz = 0;
        for (let i = 0; i < Math.min(64, pred.length); i++) if (pred[i] !== 0) nz++;
        const ppPath = usedGpu ? 'gpu-zero-copy-split' : (DIAG_FORCE_JS_POSTPROC ? 'js-cpu-split' : 'cpu-staged-split');
        log('generating', `Post-proc check: path=${ppPath} mode=kv-b1 pred[0..9]=[${Array.from(pred.slice(0, 10)).join(',')}] nonzero_in_first_64=${nz}`);
      }

      condLogitsTensor.dispose();
      uncondLogitsTensor.dispose();
    } else {
      // ── Cases A/B/C: legacy B=2 paths (monolithic / KV-B=2) ──────────────
      let inTensors;
      let ppLogitsSeqLen, ppCondLogitsOff;
      if (mainSessionIsKv && step === 0) {
        inTensors = {
          input_ids: T('int64', bIds, [2, C, maxLen]),
          audio_mask: T('bool', bMask, [2, maxLen]),
          attention_mask: T('bool', bAttn, [2, 1, maxLen, maxLen]),
          position_ids: T('int64', prefixPosIds, [2, maxLen]),
          target_positions: T('int64', prefixPosIds.slice(), [2, maxLen]),
        };
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          inTensors[`past_key_${i}`] = T('float16', prefixPastKv.zeros, prefixPastKv.shape);
          inTensors[`past_value_${i}`] = T('float16', prefixPastKv.zeros, prefixPastKv.shape);
        }
        ppLogitsSeqLen = maxLen;
        ppCondLogitsOff = targetOff;
      } else if (mainSessionIsKv) {
        for (let c = 0; c < C; c++) {
          for (let t = 0; t < numTargetTokens; t++) {
            const v = bIds[c * maxLen + targetOff + t];
            stepBIds[c * numTargetTokens + t] = v;
            stepBIds[(C + c) * numTargetTokens + t] = v;
          }
        }
        inTensors = {
          input_ids: T('int64', stepBIds, [2, C, numTargetTokens]),
          audio_mask: T('bool', stepBMask, [2, numTargetTokens]),
          attention_mask: T('bool', stepBAttn, [2, 1, numTargetTokens, maxLen]),
          position_ids: T('int64', stepPosIds, [2, numTargetTokens]),
          target_positions: T('int64', stepTargetPos, [2, numTargetTokens]),
        };
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          inTensors[`past_key_${i}`] = frozenPresentKv[i];
          inTensors[`past_value_${i}`] = frozenPresentKv[i + KV_NUM_LAYERS];
        }
        ppLogitsSeqLen = numTargetTokens;
        ppCondLogitsOff = 0;
      } else {
        inTensors = {
          input_ids: T('int64', bIds, [2, C, maxLen]),
          audio_mask: T('bool', bMask, [2, maxLen]),
          attention_mask: T('bool', bAttn, [2, 1, maxLen, maxLen]),
        };
        ppLogitsSeqLen = maxLen;
        ppCondLogitsOff = targetOff;
      }
      const results = await mainSession.run(inTensors);

      const logitsTensor = results.audio_logits;
      const canZeroCopy =
        !DIAG_FORCE_CPU_STAGED_POSTPROC &&
        !DIAG_FORCE_JS_POSTPROC &&
        gpuPostProc &&
        gpuPostProc.sharedWithOrt &&
        logitsTensor.location === 'gpu-buffer' &&
        logitsTensor.gpuBuffer;

      for (const name of Object.keys(inTensors)) {
        if (mainSessionIsKv && step > 0 && (name.startsWith('past_key_') || name.startsWith('past_value_'))) continue;
        inTensors[name].dispose();
      }

      if (mainSessionIsKv) {
        if (step === 0) {
          frozenPresentKv = new Array(2 * KV_NUM_LAYERS);
          for (let i = 0; i < KV_NUM_LAYERS; i++) {
            frozenPresentKv[i] = results[`present_key_${i}`];
            frozenPresentKv[i + KV_NUM_LAYERS] = results[`present_value_${i}`];
          }
          const locs = frozenPresentKv.map(t => t.location);
          const nGpu = locs.filter(l => l === 'gpu-buffer').length;
          log('generating', `KV diag: present_* locations -> gpu-buffer=${nGpu}/${locs.length}, logits=${logitsTensor.location}, S_full=${maxLen}, S_new=${numTargetTokens}`);
        } else {
          for (let i = 0; i < KV_NUM_LAYERS; i++) {
            results[`present_key_${i}`].dispose();
            results[`present_value_${i}`].dispose();
          }
        }
      }

      totalModelMs += performance.now() - modelT0;

      const ppParams = {
        C, V, numTargetTokens, maskId, guidanceScale, layerPenalty,
        maxLen: ppLogitsSeqLen, targetOff: ppCondLogitsOff,
      };

      const postProcT0 = performance.now();
      let logitsCpu = null;
      let usedGpu = false;
      if (canZeroCopy) {
        try {
          await gpuPostProc.runGpu(logitsTensor.gpuBuffer, ppParams, pred, scores);
          usedGpu = true;
        } catch (e) {
          log('warning', `Zero-copy post-proc failed (${e.message}); disabling and falling back to CPU-staged logits.`);
          gpuPostProc.sharedWithOrt = false;
        }
      }
      if (!usedGpu) {
        if (typeof logitsTensor.getData === 'function') {
          logitsCpu = await logitsTensor.getData();
        } else {
          logitsCpu = logitsTensor.data.slice();
        }
        if (DIAG_FORCE_JS_POSTPROC || !gpuPostProc) {
          cpuPostProcess(logitsCpu, C, ppLogitsSeqLen, V, numTargetTokens, ppCondLogitsOff, maskId, guidanceScale, layerPenalty, pred, scores);
        } else if (gpuPostProc) {
          await gpuPostProc.run(logitsCpu, ppParams, pred, scores);
        }
      }
      totalPostProcMs += performance.now() - postProcT0;

      if (step === 0) {
        let nz = 0;
        for (let i = 0; i < Math.min(64, pred.length); i++) if (pred[i] !== 0) nz++;
        const ppPath = usedGpu ? 'gpu-zero-copy' : (DIAG_FORCE_JS_POSTPROC ? 'js-cpu' : 'cpu-staged');
        log('generating', `Post-proc check: path=${ppPath} mode=${mainSessionIsKv ? 'kv' : 'mono'} pred[0..9]=[${Array.from(pred.slice(0, 10)).join(',')}] nonzero_in_first_64=${nz}`);
      }

      logitsTensor.dispose();
    }

    const sampleT0 = performance.now();
    const bigMaskId = BigInt(maskId);
    if (posTemp > 0) {
      const invTemp = 1 / posTemp;
      for (let i = 0; i < nPos; i++) {
        if (tokens[i] !== bigMaskId) { scores[i] = -Infinity; continue; }
        scores[i] = scores[i] * invTemp + (-Math.log(-Math.log(rng() + 1e-10) + 1e-10));
      }
    } else {
      for (let i = 0; i < nPos; i++)
        if (tokens[i] !== bigMaskId) scores[i] = -Infinity;
    }

    topKUnmask(scores, pred, tokens, nPos, k);

    // Update batch inputs. We always mirror new target tokens back into
    // the monolithic `bIds` layout (even in KV mode) so the step-mode
    // input_ids builder at the top of the next iteration can read them
    // uniformly.
    for (let c = 0; c < C; c++)
      for (let t = 0; t < numTargetTokens; t++) {
        const v = tokens[c * numTargetTokens + t];
        bIds[c * maxLen + targetOff + t] = v;
        bIds[(C + c) * maxLen + t] = v;
      }
    totalSampleMs += performance.now() - sampleT0;

    const stepMs = performance.now() - stepT0;
    totalInferenceMs += stepMs;
    log('generating', `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)`);

    if (step % 5 === 0 && globalThis.gc) {
      globalThis.gc();
    }
  }

  // Release frozen-prefix GPU buffers now that the loop is done. Without this
  // each utterance would leak ~1.1 GB of VRAM in B=2 mode (28 layers × 2 ×
  // 2B × 8 × maxLen × 128); in B=1 split mode we leak ~700 MB cond + ~300 MB
  // uncond (uncond's S_full is smaller).
  if (frozenPresentKv) {
    for (const t of frozenPresentKv) { try { t.dispose(); } catch (_e) {} }
    frozenPresentKv = null;
  }
  if (frozenPresentKvCond) {
    for (const t of frozenPresentKvCond) { try { t.dispose(); } catch (_e) {} }
    frozenPresentKvCond = null;
  }
  if (frozenPresentKvUncond) {
    for (const t of frozenPresentKvUncond) { try { t.dispose(); } catch (_e) {} }
    frozenPresentKvUncond = null;
  }

  const modeLabel = mainSessionIsKvB1
    ? 'kv-b1-split'
    : (mainSessionIsKv ? 'kv-cached' : 'monolithic');
  const uncondSuffix = mainSessionIsKvB1
    ? ` | uncond runs: ${uncondRunsExecuted}/${numStep}`
    : '';
  const otherMs = Math.max(0, totalInferenceMs - totalModelMs - totalPostProcMs - totalSampleMs);
  log('perf', `${numStep} steps in ${totalInferenceMs.toFixed(0)}ms total | model: ${totalModelMs.toFixed(0)}ms (${(totalModelMs / Math.max(1, numStep)).toFixed(0)}ms/step) | postproc+sync: ${totalPostProcMs.toFixed(0)}ms | sample: ${totalSampleMs.toFixed(0)}ms | other: ${otherMs.toFixed(0)}ms | mode: ${modeLabel} | variant: ${mainSessionVariant}${uncondSuffix}`);

  return tokens;
}

// ─── Decode tokens to audio ─────────────────────────────────────────────────

async function decodeTokens(tokens, C, T) {
  log('decoding', 'Converting tokens to audio...');
  const codes = new BigInt64Array(C * T);
  codes.set(tokens);
  const inTensor = new ort.Tensor('int64', codes, [1, C, T]);
  const r = await decoderSession.run({ audio_codes: inTensor });
  let outData;
  if (typeof r.audio_values.getData === 'function') {
    outData = await r.audio_values.getData();
  } else {
    outData = r.audio_values.data;
  }
  inTensor.dispose();
  r.audio_values.dispose();
  return outData;
}

function audioTokensToRows(tokens, C, T) {
  const rows = [];
  for (let c = 0; c < C; c++) {
    const row = [];
    for (let t = 0; t < T; t++) row.push(Number(tokens[c * T + t]));
    rows.push(row);
  }
  return rows;
}

// ─── Encode reference audio ─────────────────────────────────────────────────

async function encodeRefAudio(pcmF32) {
  if (!encoderSession) {
    log('warning', 'Encoder session not available — voice cloning disabled');
    return null;
  }
  log('encoding', 'Encoding reference audio...');
  const hopLength = 960;
  const clipLen = pcmF32.length - (pcmF32.length % hopLength);
  const aligned = pcmF32.slice(0, clipLen);
  const inputTensor = new ort.Tensor('float32', aligned, [1, 1, aligned.length]);
  const encResult = await encoderSession.run({ input_values: inputTensor });
  const codesData = encResult.audio_codes.data;
  const codeDims = encResult.audio_codes.dims;
  const T = Number(codeDims[2]);
  const refAudioTokens = [];
  for (let c = 0; c < 8; c++) {
    const row = [];
    for (let t = 0; t < T; t++) row.push(Number(codesData[c * T + t]));
    refAudioTokens.push(row);
  }

  inputTensor.dispose();
  encResult.audio_codes.dispose();

  log('encoding', `Encoded: ${T} tokens (${(aligned.length / config.sampling_rate).toFixed(1)}s)`);
  return refAudioTokens;
}

// ─── Init — load models and tokenizer ───────────────────────────────────────

async function init(modelBaseUrl = DEFAULT_MODEL_BASE_URL) {
  try {
    // Detect WebGPU. We do NOT pre-create a device — ORT Web 1.20 creates its
    // own internal GPUDevice when the first WebGPU session is built, and on
    // this build it silently ignores a preset ort.env.webgpu.device. Binding
    // ORT's output GPU buffer into a pipeline on a *different* device is a
    // WebGPU validation error ("Buffer … cannot be used with [Device]").
    // Instead we create the session first, then grab ort.env.webgpu.device
    // and hand that same device to GpuPostProcessor so cross-device binding
    // never happens.
    let hasWorkingGPU = false;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        hasWorkingGPU = !!adapter;
        if (adapter) {
          try {
            const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
            log('init', `WebGPU adapter: ${info.vendor || '?'} ${info.device || '?'} (${info.description || '?'})`);
          } catch (e) {
            log('init', 'WebGPU adapter found (could not read info)');
          }
        }
      } catch (e) {
        log('init', `WebGPU probe failed: ${e.message}`);
      }
    }
    if (!hasWorkingGPU) {
      log('init', 'No WebGPU — will use WASM fallback (slower)');
    }

    // Provide modelBaseUrl for lazy loading
    window._modelBaseUrl = modelBaseUrl;

    // Load config
    log('loading', 'Loading config...');
    config = await (await fetch(`${modelBaseUrl}/omnivoice-config.json`)).json();
    log('loading', `Config loaded: ${config.num_audio_codebook} codebooks, vocab ${config.audio_vocab_size}, sr ${config.sampling_rate}`);

    // Load tokenizer
    log('loading', `Loading tokenizer (Qwen2 BPE) from ${DEFAULT_MODEL_REPO_ID}...`);
    tokenizer = await AutoTokenizer.from_pretrained(DEFAULT_MODEL_REPO_ID);
    log('loading', 'Tokenizer loaded.');

    // Whisper ASR is now lazy-loaded natively when new voices require pre-computation.

    const mainModelFile = MAIN_MODEL_FILE;
    const manifestFile = MAIN_MODEL_MANIFEST;
    mainSessionIsKv = true;
    mainSessionIsKvB1 = true;
    mainSessionVariant = MAIN_MODEL_VARIANT;

    log('loading', `Using production main-model variant: ${mainSessionVariant}.`);

    // Load model data shards
    log('downloading', `Loading main model shards into memory (${manifestFile})...`);
    const dataFiles = await (await fetch(`${modelBaseUrl}/${manifestFile}`)).json();

    let externalData = [];
    for (let i = 0; i < dataFiles.length; i++) {
      const fname = dataFiles[i];
      log('downloading', `Shard ${i + 1}/${dataFiles.length}: ${fname}...`);
      const resp = await fetch(`${modelBaseUrl}/${fname}`);
      if (!resp.ok) throw new Error(`Failed to fetch ${fname}: ${resp.status}`);
      const buf = await resp.arrayBuffer();
      externalData.push({ path: fname, data: new Uint8Array(buf) });
      log('downloading', `Shard ${i + 1}/${dataFiles.length}: ${(buf.byteLength / 1e6).toFixed(0)} MB loaded`);
      if (globalThis.gc) globalThis.gc();
    }

    // Create ONNX sessions
    let actualBackend = 'cpu';
    log('loading', 'Creating main model session (WebGPU)...');

    // We strictly ban contiguous pool arenas to save gigabytes of shared RAM VRAM.
    // preferredOutputLocation keeps `audio_logits` on the GPU as a GPUBuffer so
    // our GpuPostProcessor can bind it directly — no 25 MB GPU→CPU→GPU bounce
    // per diffusion step. This is safe only if we can then access ORT's
    // internal device to share it with GpuPostProcessor (see below).
    // In KV mode we also keep every present_key_i / present_value_i on the GPU
    // so the step-0 prefix call's frozen-prefix buffers never transit CPU.
    const outLoc = {};
    if (!DIAG_FORCE_CPU_STAGED_POSTPROC && !DIAG_FORCE_JS_POSTPROC) {
      outLoc.audio_logits = 'gpu-buffer';
    }
    if (mainSessionIsKv) {
      if (!DIAG_FORCE_CPU_KV_CACHE) {
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          outLoc[`present_key_${i}`] = 'gpu-buffer';
          outLoc[`present_value_${i}`] = 'gpu-buffer';
        }
      }
    }
    const opt = {
      externalData: externalData,
      graphOptimizationLevel: 'all',
      enableMemPattern: false,
      enableCpuMemArena: false,
      preferredOutputLocation: outLoc,
    };

    if (!hasWorkingGPU) {
      throw new Error("WebGPU is not available. The program requires WebGPU to run.");
    }

    try {
      opt.executionProviders = ['webgpu'];
      mainSession = await ort.InferenceSession.create(`${modelBaseUrl}/${mainModelFile}`, opt);
      actualBackend = 'webgpu';
    } catch (e) {
      log('error', `Main model WebGPU failed: ${e.message}`);
      throw new Error(`WebGPU InferenceSession creation failed: ${e.message}`);
    }

    // CRITICAL: Free the 2.5GB of JavaScript ArrayBuffers so the browser doesn't swap + lock up!
    log('loading', 'Freeing JS shard memory buffers...');
    externalData = null;
    opt.externalData = null;
    if (globalThis.gc) {
      globalThis.gc();
      log('loading', 'Forced Garbage Collection');
    }

    log('init', `Main model backend: ${actualBackend}; variant: ${mainSessionVariant}; file: ${mainModelFile}`);

    if (actualBackend === 'webgpu') {
      // Grab ORT's internal device so GpuPostProcessor can bind ORT's output
      // GPU buffers directly. If ORT doesn't expose it, fall back to an
      // isolated device and we'll stage logits through CPU (still correct,
      // just not zero-copy).
      let ortDevice = null;
      try { ortDevice = ort.env && ort.env.webgpu && ort.env.webgpu.device; }
      catch (_e) { ortDevice = null; }

      gpuPostProc = new GpuPostProcessor(ortDevice);
      await gpuPostProc.init();
      const diagFlags = [
        DIAG_FORCE_CPU_STAGED_POSTPROC ? 'cpu-staged-postproc' : null,
        DIAG_FORCE_JS_POSTPROC ? 'js-postproc' : null,
        DIAG_FORCE_WASM_DECODER ? 'wasm-decoder' : null,
        DIAG_FORCE_CPU_KV_CACHE ? 'cpu-kv-cache' : null,
      ].filter(Boolean).join(', ') || 'none';
      log('init', `Initialized GpuPostProcessor (zero-copy logits: ${ortDevice ? 'on' : 'off — ORT device not accessible'}; diagnostics: ${diagFlags})`);
    }

    // Decoder on WebGPU. Historically this was pinned to WASM because the stock
    // Encodec decoder produced heavily-quantized/silent audio. The actual cause
    // was not a Vulkan driver bug — it was ORT-Web WebGPU EP issues:
    //   1. `output_padding` on ConvTranspose is silently dropped.
    //   2. ConvTranspose(k=6, s=3, pads=[2,2]) (block.4/conv_t1) produces
    //      ~half-magnitude output regardless of input size.
    //   3. ORT-Web 1.24+ regressed the early per-codebook Gather/MatMul
    //      projection stack, causing intelligible-but-corrupted audio.
    // These are worked around at the graph level by `decoder_fix_output_padding.py`
    // and `decoder_fuse_codebook_project.py` in decoder_webgpu_fix/, which write
    // `omnivoice-decoder-webgpu.onnx` next to the stock decoder with equivalent
    // replacements. Verified against the CPU reference by `decoder_gpu_bisect.py`.
    // We try the patched model on WebGPU first and transparently fall back to the
    // stock decoder on WASM if either the file or the WebGPU session can't be built.
    let decEp = ['webgpu'];
    let decoderLoaded = false;
    try {
      if (DIAG_FORCE_WASM_DECODER) throw new Error('DIAG_FORCE_WASM_DECODER=true');
      log('loading', 'Trying WebGPU-patched decoder (omnivoice-decoder-webgpu.onnx, ~111MB)...');
      const decResp = await fetch(`${modelBaseUrl}/omnivoice-decoder-webgpu.onnx`);
      if (!decResp.ok) throw new Error(`fetch status ${decResp.status}`);
      const decBuf = await decResp.arrayBuffer();
      decoderSession = await ort.InferenceSession.create(new Uint8Array(decBuf), { executionProviders: decEp, enableMemPattern: false, enableCpuMemArena: false });
      decoderLoaded = true;
      log('loading', 'Decoder session created on WebGPU.');
    } catch (e) {
      log('loading', `WebGPU-patched decoder unavailable (${e.message}) — falling back to stock decoder on WASM.`);
    }
    if (!decoderLoaded) {
      decEp = ['wasm'];
      log('loading', 'Downloading omnivoice-decoder.onnx (~87MB)...');
      try {
        const decResp = await fetch(`${modelBaseUrl}/omnivoice-decoder.onnx`);
        const decBuf = await decResp.arrayBuffer();
        log('loading', 'Creating decoder session (wasm)...');
        decoderSession = await ort.InferenceSession.create(new Uint8Array(decBuf), { executionProviders: decEp, enableMemPattern: false, enableCpuMemArena: false });
      } catch (e) {
        log('error', `Failed to load decoder: ${e.message}`);
        throw e;
      }
    }
    window._decEp = decEp;

    // Pre-load encoder for voice cloning (654MB). The encoder was previously
    // pinned to WASM out of caution, but bisection (encoder_webgpu_fix/) shows
    // the stock graph runs bit-accurately on the ORT-Web WebGPU EP on BC-250:
    //   - audio_codes matches CPU exactly at every tested input size
    //   - no op (Conv, LayerNorm, InstanceNorm, Softmax, MatMul, ArgMax)
    //     diverges beyond normal fp32 noise (rel_l2 < 1e-5)
    // Running on WebGPU is ~4x faster than WASM on this hardware, so we try
    // WebGPU first and transparently fall back to WASM if session creation
    // fails for any reason (e.g. an older driver that rejects the model).
    log('downloading', 'Downloading encoder.onnx (654MB)...');
    try {
      const encResp = await fetch(`${modelBaseUrl}/omnivoice-encoder-fixed.onnx`);
      const encBuf = await encResp.arrayBuffer();
      const encData = new Uint8Array(encBuf);
      try {
        log('loading', 'Creating encoder session (WebGPU)...');
        encoderSession = await ort.InferenceSession.create(encData, {
          executionProviders: ['webgpu'],
          enableMemPattern: false,
          enableCpuMemArena: false,
        });
        window._encEp = 'webgpu';
        log('init', 'Encoder pre-loaded on WebGPU.');
      } catch (e) {
        log('warning', `Encoder WebGPU session failed (${e.message}); falling back to WASM.`);
        encoderSession = await ort.InferenceSession.create(encData, {
          executionProviders: ['wasm'],
          enableMemPattern: false,
          enableCpuMemArena: false,
        });
        window._encEp = 'wasm';
        log('init', 'Encoder pre-loaded on WASM (fallback).');
      }
      window._hasEncoderWarmedUp = true;
    } catch (e) {
      log('error', `Failed to pre-load encoder: ${e.message}`);
    }

    // Warm up sessions
    log('loading', 'Warming up sessions...');
    try {
      if (mainSessionIsKv) {
        // KV graph: build dummy prefix-style call (S_new == S_full == 4).
        // The batch dim differs between B=1 and B=2 graphs; everything else is
        // identical (the trace uses dynamic axes so the graph accepts either).
        const B = mainSessionIsKvB1 ? 1 : 2;
        const S = 4;
        const dummyIds = new BigInt64Array(B * 8 * S).fill(1024n);
        const dummyMask = new Uint8Array(B * S);
        const dummyAttn = new Uint8Array(B * 1 * S * S).fill(1);
        const dummyPos = new BigInt64Array(B * S);
        for (let b = 0; b < B; b++) for (let s = 0; s < S; s++) dummyPos[b * S + s] = BigInt(s);
        const feed = {
          input_ids: new ort.Tensor('int64', dummyIds, [B, 8, S]),
          audio_mask: new ort.Tensor('bool', dummyMask, [B, S]),
          attention_mask: new ort.Tensor('bool', dummyAttn, [B, 1, S, S]),
          position_ids: new ort.Tensor('int64', dummyPos, [B, S]),
          target_positions: new ort.Tensor('int64', dummyPos.slice(), [B, S]),
        };
        const kvSize = B * KV_HEADS * S * KV_HEAD_DIM;
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          feed[`past_key_${i}`] = new ort.Tensor('float16', new Float16Array(kvSize), [B, KV_HEADS, S, KV_HEAD_DIM]);
          feed[`past_value_${i}`] = new ort.Tensor('float16', new Float16Array(kvSize), [B, KV_HEADS, S, KV_HEAD_DIM]);
        }
        const warm = await mainSession.run(feed);
        for (const k in warm) warm[k].dispose();
        for (const k in feed) feed[k].dispose();
      } else {
        const dummyIds = new BigInt64Array(2 * 8 * 4).fill(1024n);
        const dummyMask = new Uint8Array(2 * 4);
        const dummyAttn = new Uint8Array(2 * 1 * 4 * 4).fill(1);
        await mainSession.run({
          input_ids: new ort.Tensor('int64', dummyIds, [2, 8, 4]),
          audio_mask: new ort.Tensor('bool', dummyMask, [2, 4]),
          attention_mask: new ort.Tensor('bool', dummyAttn, [2, 1, 4, 4]),
        });
      }
      const dummyCodes = new BigInt64Array(8 * 2).fill(0n);
      await decoderSession.run({ audio_codes: new ort.Tensor('int64', dummyCodes, [1, 8, 2]) });
      if (window._hasEncoderWarmedUp) {
        const dummyAudio = new Float32Array(960);
        await encoderSession.run({ input_values: new ort.Tensor('float32', dummyAudio, [1, 1, 960]) });
      }
    } catch (e) {
      log('init', `Warm-up error (non-fatal): ${e.message}`);
    }

    log('ready', `Engine ready. Backend: ${actualBackend}`);
    return { backend: actualBackend, config };
  } catch (err) {
    log('error', `Init failed: ${err.message}\n${err.stack}`);
    throw err;
  }
}

// ─── Synthesize ─────────────────────────────────────────────────────────────

async function synthesize(params) {
  const {
    text, lang = null, refAudio = null, refText = null, instruct = null,
    numStep = 20, guidanceScale = 4.0, tShift = 0.05, speed = 1.0,
    seed = null, denoise = true, precalculatedRefTokens = null,
    returnAudioTokens = false
  } = params;

  try {
    // Use seeded PRNG for deterministic output when seed is provided
    rng = seed != null ? mulberry32(seed) : Math.random;

    // Encode reference audio for voice cloning if provided
    let refAudioTokens = precalculatedRefTokens;
    let autoRefText = refText;

    if (!refAudioTokens && refAudio) {
      if (!autoRefText || autoRefText.trim() === '') {
        log('encoding', 'No reference text provided. Using pre-loaded Whisper ASR to transcribe audio...');
        await ensureAsrModel();
        // Resample 24kHz Float32Array down to 16kHz for Whisper
        const inData = new Float32Array(refAudio);
        const ratio = 24000 / 16000;
        const outLen = Math.floor(inData.length / ratio);
        const whisperAudio = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
          const srcPos = i * ratio;
          const idx = Math.floor(srcPos);
          const frac = srcPos - idx;
          if (idx + 1 < inData.length) {
            whisperAudio[i] = inData[idx] * (1 - frac) + inData[idx + 1] * frac;
          } else {
            whisperAudio[i] = inData[inData.length - 1];
          }
        }

        const asrResult = await asrModel(whisperAudio);
        autoRefText = asrResult.text.trim();
        log('encoding', `Transcribed reference text: "${autoRefText}"`);
      }

      if (!encoderSession) {
        throw new Error('Encoder session not loaded. Check init() logs.');
      }
      const pcmF32 = new Float32Array(refAudio);
      refAudioTokens = await encodeRefAudio(pcmF32);
    }

    // Duration estimation
    const estRefText = autoRefText || 'Nice to meet you.';
    const estRefTokens = refAudioTokens ? refAudioTokens[0].length : 25;
    const numTargetTokens = estimateTargetTokens(text, estRefText, estRefTokens, speed);

    log('preparing', `Target: ${numTargetTokens} tokens, ${numStep} steps`);

    const inputs = await prepareInferenceInputs(text, numTargetTokens, tokenizer, config, {
      lang, instruct, refText: autoRefText, refAudioTokens, denoise,
    });

    const tokens = await generateIterative(inputs, config, numStep, guidanceScale, tShift);

    // DEBUG: print first 20 tokens to see if they unmasked properly
    log('decoding', `First 20 tokens (cb0): [${Array.from(tokens.slice(0, 20)).join(', ')}]`);

    const pcm = await decodeTokens(tokens, config.num_audio_codebook, numTargetTokens);

    log('done', `Generated ${pcm.length} samples (${(pcm.length / config.sampling_rate).toFixed(1)}s at ${config.sampling_rate}Hz)`);

    // If a resultUrl was provided, POST the raw Float32 PCM bytes to the
    // local HTTP server instead of JSON-serializing them through CDP.
    // This avoids the ~1s/chunk overhead of JSON-encoding tens of thousands
    // of floats through the Chrome DevTools Protocol.
    const metadata = {
      sampleRate: config.sampling_rate,
      numTokens: numTargetTokens,
      pcmLength: pcm.length,
    };
    if (returnAudioTokens) {
      metadata.audioTokens = audioTokensToRows(tokens, config.num_audio_codebook, numTargetTokens);
    }

    if (params.resultUrl) {
      const blob = new Blob([pcm.buffer], { type: 'application/octet-stream' });
      await fetch(params.resultUrl, { method: 'POST', body: blob });
      return metadata;
    }

    // Fallback: return as plain array for JSON serialization back to Python
    return {
      pcm: Array.from(pcm),
      ...metadata,
    };
  } catch (err) {
    log('error', `Synthesis failed: ${err.message}\n${err.stack}`);
    throw err;
  }
}

// ─── Direct Voice Artifact Generator ────────────────────────────────────────

async function generateVoiceData(pcmF32, hint) {
  let autoRefText = hint || '';
  let usedAsr = false;
  if (!autoRefText || autoRefText.trim() === '') {
    log('encoding', 'Transcribing voice for persistent local storage...');
    const ratio = 24000 / 16000;
    const outLen = Math.floor(pcmF32.length / ratio);
    const whisperAudio = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcPos = i * ratio;
      const idx = Math.floor(srcPos);
      const frac = srcPos - idx;
      if (idx + 1 < pcmF32.length) {
        whisperAudio[i] = pcmF32[idx] * (1 - frac) + pcmF32[idx + 1] * frac;
      } else {
        whisperAudio[i] = pcmF32[pcmF32.length - 1];
      }
    }

    await ensureAsrModel();
    const asrResult = await asrModel(whisperAudio);
    autoRefText = asrResult.text.trim();
    usedAsr = true;
  }

  if (!encoderSession) {
    throw new Error('Encoder session not loaded. Cannot precalculate tokens.');
  }

  const refAudioTokens = await encodeRefAudio(pcmF32);
  if (usedAsr) {
    unloadAsrModel();
  }
  return { refText: autoRefText, tokens: refAudioTokens };
}

// ─── Expose on window for Playwright ────────────────────────────────────────

window._abortRequested = false;
window.ttsEngine = {
  init,
  synthesize,
  generateVoiceData,
  abortGeneration: () => { window._abortRequested = true; }
};
window.ttsReady = false;

log('boot', 'TTS engine module loaded, waiting for init()...');/**
 * GPU post-processing for TTS diffusion loop.
 * Moves log-softmax + CFG fusion + argmax from JS CPU to WebGPU compute shaders.
 */

// ── FP16 logits post-processing shader ───────────────────────────────────────
//
// The logits tensor is bound as `array<u32>` and decoded at load-time via
// `unpack2x16float` (core WGSL, no feature extension). This halves the bytes
// streamed from VRAM per diffusion step vs. fp32 and removes a graph-boundary
// Cast(fp16→fp32) in the ONNX head — see optimization #7 in README.md.
//
// Layout: one thread per (codebook, target-token) position. Each thread does
// its own log-softmax → CFG fusion → argmax over V serially. No workgroup
// cooperation or shared memory; intentionally simple since the measured
// post-proc overhead (~20 ms/step) is small relative to the model forward
// pass (~80–100 ms/step). A cooperative-reduction version was tried and
// reverted (see README #7) — net end-to-end impact was in the noise but it
// added ~130 lines of shader that nothing else in the engine benefits from.
//
// V=1025 is odd; total element count (2·C·S·V) is still even because the
// leading factor 2 (cond+uncond) is even, so the u32 array length is exactly
// (elements / 2) with no tail alignment issue. Per-element addressing:
//   u32_idx = fp16_idx >> 1,   lane = fp16_idx & 1
//   lane==0 → unpack2x16float(packed).x   (low 16 bits)
//   lane==1 → unpack2x16float(packed).y   (high 16 bits)
// The shader takes TWO logits storage bindings (cond + uncond) and per-side
// stride parameters. This single WGSL serves both:
//
//   1. B=2 monolithic / B=2 KV: both bindings point to the same packed
//      (2, C, maxLen, V) logits buffer. Cond addressing starts at 0; uncond
//      addressing starts at C*maxLen*V (via `uncondBatchOff`).
//   2. B=1 KV split: bindings point to two independent
//      (1, C, S_cond, V) and (1, C, S_uncond, V) logits buffers from two
//      separate mainSession.run() calls per step. `uncondBatchOff` is 0.
//
// Keeping one shader means zero extra pipeline state to manage. The B=2 path
// ends up doing redundant identical fp16 loads via the two bindings, but the
// L1 cache absorbs this — there's no measurable penalty vs the previous
// one-binding version.
const WGSL = /* wgsl */ `

// CFG + argmax post-processing shader.
// Input:  cond/uncond logits buffers (fp16, packed as u32 pairs).
// Output: pred[nPos]   — argmax vocab id per position
//         scoresBuf[nPos] — log-prob of that argmax minus codebook penalty
//
// The monolithic (B=2) and split (B=1) zero-copy paths both use this shader;
// they differ only in how they set up the cond/uncond bindings and the
// uncondBatchOff param.

struct Params {
  C:               u32,  // num codebooks (8)
  V:               u32,  // vocab size (1025)
  numTargetTokens: u32,
  maskId:          u32,
  condMaxLen:      u32,  // cond logits seq-dim
  condTargetOff:   u32,  // offset into cond seq-dim where target region begins
  uncondMaxLen:    u32,  // uncond logits seq-dim
  uncondTargetOff: u32,  // offset into uncond seq-dim where target region begins
  uncondBatchOff:  u32,  // additional offset (in units of seq_len*V) prepended
                         //   to every uncond index. Used in the B=2 path to
                         //   skip past the cond rows in the shared buffer
                         //   (= C * maxLen). Zero in the B=1 split path.
  guidanceScale:   f32,
  layerPenalty:    f32,
};

@group(0) @binding(0) var<uniform> p : Params;
@group(0) @binding(1) var<storage, read>       logitsCondU32   : array<u32>;
@group(0) @binding(2) var<storage, read>       logitsUncondU32 : array<u32>;
@group(0) @binding(3) var<storage, read_write> pred            : array<i32>;
@group(0) @binding(4) var<storage, read_write> scoresBuf       : array<f32>;

fn load_fp16_cond(i: u32) -> f32 {
  let packed = logitsCondU32[i >> 1u];
  let pair = unpack2x16float(packed);
  if ((i & 1u) == 0u) { return pair.x; }
  return pair.y;
}

fn load_fp16_uncond(i: u32) -> f32 {
  let packed = logitsUncondU32[i >> 1u];
  let pair = unpack2x16float(packed);
  if ((i & 1u) == 0u) { return pair.x; }
  return pair.y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let posIdx = gid.x;
  let N = p.numTargetTokens;
  let nPos = p.C * N;
  if (posIdx >= nPos) { return; }

  let c = posIdx / N;
  let t = posIdx % N;
  let V = p.V;
  let cBase = (c * p.condMaxLen + p.condTargetOff + t) * V;
  let uBase = (p.uncondBatchOff + c * p.uncondMaxLen + p.uncondTargetOff + t) * V;
  let NEG_INF : f32 = -1.0e30;
  let gScale1 = 1.0 + p.guidanceScale;

  var cMax : f32 = NEG_INF;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    cMax = max(cMax, load_fp16_cond(cBase + v));
  }
  var cSum : f32 = 0.0;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    cSum = cSum + exp(load_fp16_cond(cBase + v) - cMax);
  }
  let cLse = cMax + log(cSum);

  var uMax : f32 = NEG_INF;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    uMax = max(uMax, load_fp16_uncond(uBase + v));
  }
  var uSum : f32 = 0.0;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    uSum = uSum + exp(load_fp16_uncond(uBase + v) - uMax);
  }
  let uLse = uMax + log(uSum);

  var gMax : f32 = NEG_INF;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    let gv = gScale1 * (load_fp16_cond(cBase + v) - cLse) - p.guidanceScale * (load_fp16_uncond(uBase + v) - uLse);
    gMax = max(gMax, gv);
  }
  var gSum : f32 = 0.0;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    let gv = gScale1 * (load_fp16_cond(cBase + v) - cLse) - p.guidanceScale * (load_fp16_uncond(uBase + v) - uLse);
    gSum = gSum + exp(gv - gMax);
  }
  let gLse = gMax + log(gSum);

  var bestV : u32 = 0u;
  var bestS : f32 = NEG_INF;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    if (v == p.maskId) { continue; }
    let gv = gScale1 * (load_fp16_cond(cBase + v) - cLse) - p.guidanceScale * (load_fp16_uncond(uBase + v) - uLse);
    let lp = gv - gLse;
    if (lp > bestS) { bestS = lp; bestV = v; }
  }

  pred[posIdx] = i32(bestV);
  scoresBuf[posIdx] = bestS - p.layerPenalty * f32(c);
}
`;

class GpuPostProcessor {
  constructor(device = null) {
    this.device = device;         // if provided, shared with ORT (zero-copy path)
    this._ownsDevice = !device;
    this.sharedWithOrt = !!device; // consumers check this before attempting zero-copy
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.logitsBuf = null;        // only allocated in the legacy CPU-staged path
    this.paramsBuf = null;
    this.predBuf = null;
    this.scoresBuf = null;
    this.predReadBuf = null;
    this.scoresReadBuf = null;
    this._logitsSize = 0;
    this._nPos = 0;
  }

  async init() {
    if (!this.device) {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      this.device = await adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });
      this._ownsDevice = true;
    }

    const module = this.device.createShaderModule({ code: WGSL });
    if (typeof module.getCompilationInfo === 'function') {
      try {
        const info = await module.getCompilationInfo();
        if (info && info.messages && info.messages.length) {
          for (const m of info.messages) {
            log('post-proc', `shader ${m.type}@L${m.lineNum}:${m.linePos}: ${m.message}`);
          }
        }
      } catch (_e) { /* non-fatal */ }
    }
    this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.device.pushErrorScope('validation');
    this.pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: 'main' },
    });
    const pipelineErr = await this.device.popErrorScope();
    if (pipelineErr) {
      log('error', `GpuPostProcessor pipeline invalid: ${pipelineErr.message}`);
      throw new Error(`post-proc pipeline creation failed: ${pipelineErr.message}`);
    }

    // Params struct: 11 u32/f32 = 44 bytes, rounded up to 48 (WebGPU min
    // 16-byte alignment on uniform buffer binding offsets).
    this.paramsBuf = this.device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // Serialize the 12-entry Params struct into the paramsBuf. `ext` carries
  // the new split-aware fields; unsupplied fields fall back to B=2 defaults
  // derived from the legacy `params` object so old callers work untouched.
  _writeParams(params, ext) {
    const { C, V, numTargetTokens, maskId, guidanceScale, layerPenalty } = params;
    // Legacy params used `maxLen` (cond seq-dim) + `targetOff` (cond off).
    const condMaxLen = ext.condMaxLen ?? params.maxLen;
    const condTargetOff = ext.condTargetOff ?? params.targetOff;
    const uncondMaxLen = ext.uncondMaxLen ?? params.maxLen;
    const uncondTargetOff = ext.uncondTargetOff ?? 0;
    const uncondBatchOff = ext.uncondBatchOff ?? (C * condMaxLen);
    const paramData = new ArrayBuffer(48);
    const u32 = new Uint32Array(paramData);
    const f32 = new Float32Array(paramData);
    u32[0] = C;
    u32[1] = V;
    u32[2] = numTargetTokens;
    u32[3] = maskId;
    u32[4] = condMaxLen;
    u32[5] = condTargetOff;
    u32[6] = uncondMaxLen;
    u32[7] = uncondTargetOff;
    u32[8] = uncondBatchOff;
    f32[9] = guidanceScale;
    f32[10] = layerPenalty;
    // u32[11] is padding; WebGPU requires uniform buffer size to be a
    // multiple of 16 bytes. Not read by the shader.
    this.device.queue.writeBuffer(this.paramsBuf, 0, paramData);
  }

  /** Grow pred/scores output buffers for a new synthesis run. The logits
   *  staging buffer is only needed on the CPU-staged legacy path and is
   *  allocated lazily in run().
   */
  prepare(C, maxLen, V, numTargetTokens) {
    const dev = this.device;
    const nPos = C * numTargetTokens;
    const posBytes = nPos * 4;

    if (!this.predBuf || nPos > this._nPos) {
      if (this.predBuf) this.predBuf.destroy();
      if (this.scoresBuf) this.scoresBuf.destroy();
      if (this.predReadBuf) this.predReadBuf.destroy();
      if (this.scoresReadBuf) this.scoresReadBuf.destroy();

      this.predBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.scoresBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.predReadBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this.scoresReadBuf = dev.createBuffer({
        size: posBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this._nPos = nPos;
    }
  }

  _ensureLogitsBuf(logitsBytes) {
    if (!this.logitsBuf || logitsBytes > this._logitsSize) {
      if (this.logitsBuf) this.logitsBuf.destroy();
      this.logitsBuf = this.device.createBuffer({
        size: logitsBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this._logitsSize = logitsBytes;
    }
  }

  /**
   * Zero-copy path, B=2 mode: ORT owns a single packed (2, C, maxLen, V)
   * logits GPU buffer; we bind it to both cond AND uncond bindings and rely
   * on `uncondBatchOff = C*maxLen` to skip into the uncond rows. Caller must
   * not dispose the ORT tensor until this promise resolves.
   */
  async runGpu(logitsGpuBuffer, params, predOut, scoresOut) {
    const dev = this.device;
    const { C, maxLen, V, numTargetTokens } = params;
    const nPos = C * numTargetTokens;
    // logits are fp16 (2 B/elem) at the graph boundary.
    const logitsBytes = 2 * C * maxLen * V * 2;

    this._writeParams(params, {});  // B=2 defaults: uncondBatchOff=C*maxLen

    const bindGroup = dev.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: logitsGpuBuffer, size: logitsBytes } },
        { binding: 2, resource: { buffer: logitsGpuBuffer, size: logitsBytes } },
        { binding: 3, resource: { buffer: this.predBuf, size: nPos * 4 } },
        { binding: 4, resource: { buffer: this.scoresBuf, size: nPos * 4 } },
      ],
    });

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(nPos / 64));
    pass.end();
    encoder.copyBufferToBuffer(this.predBuf, 0, this.predReadBuf, 0, nPos * 4);
    encoder.copyBufferToBuffer(this.scoresBuf, 0, this.scoresReadBuf, 0, nPos * 4);
    dev.queue.submit([encoder.finish()]);

    await Promise.all([
      this.predReadBuf.mapAsync(GPUMapMode.READ),
      this.scoresReadBuf.mapAsync(GPUMapMode.READ),
    ]);
    predOut.set(new Int32Array(this.predReadBuf.getMappedRange(), 0, nPos));
    scoresOut.set(new Float32Array(this.scoresReadBuf.getMappedRange(), 0, nPos));
    this.predReadBuf.unmap();
    this.scoresReadBuf.unmap();
  }

  /**
   * Zero-copy path, B=1 split mode: cond and uncond logits come from two
   * independent mainSession.run() calls. We bind each buffer to its own slot
   * and tell the shader `uncondBatchOff = 0` so both use pure row-major
   * indexing into their own buffer.
   *
   * `ext` must specify condMaxLen/condTargetOff/uncondMaxLen/uncondTargetOff
   * and leave uncondBatchOff as 0 (the default when omitted here is handled
   * in _writeParams, but the B=2 default assumes a shared buffer; we pass
   * 0 explicitly).
   */
  async runGpuSplit(logitsCondBuf, logitsUncondBuf, condBytes, uncondBytes, params, ext, predOut, scoresOut) {
    const dev = this.device;
    const nPos = params.C * params.numTargetTokens;

    this._writeParams(params, { ...ext, uncondBatchOff: 0 });

    // ORT may pad its GPU allocations (e.g. to a power-of-two boundary) so
    // the true backing-buffer size can be larger than `condBytes`/`uncondBytes`.
    // We omit `size` on the logits bindings and let WebGPU bind the entire
    // buffer; the shader only reads `condBytes`/`uncondBytes` worth of
    // elements via its own index math, so over-binding is harmless.
    const bindGroup = dev.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: logitsCondBuf } },
        { binding: 2, resource: { buffer: logitsUncondBuf } },
        { binding: 3, resource: { buffer: this.predBuf, size: nPos * 4 } },
        { binding: 4, resource: { buffer: this.scoresBuf, size: nPos * 4 } },
      ],
    });
    void condBytes; void uncondBytes; // kept as callsite documentation only

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(nPos / 64));
    pass.end();
    encoder.copyBufferToBuffer(this.predBuf, 0, this.predReadBuf, 0, nPos * 4);
    encoder.copyBufferToBuffer(this.scoresBuf, 0, this.scoresReadBuf, 0, nPos * 4);
    dev.queue.submit([encoder.finish()]);

    await Promise.all([
      this.predReadBuf.mapAsync(GPUMapMode.READ),
      this.scoresReadBuf.mapAsync(GPUMapMode.READ),
    ]);
    predOut.set(new Int32Array(this.predReadBuf.getMappedRange(), 0, nPos));
    scoresOut.set(new Float32Array(this.scoresReadBuf.getMappedRange(), 0, nPos));
    this.predReadBuf.unmap();
    this.scoresReadBuf.unmap();
  }

  /**
   * CPU-staged fallback: ORT-web handed us fp16 logits on CPU
   * (Float16Array on modern ort-web, Uint16Array on older — both store fp16
   * bit patterns at 2 B/elem). We upload the raw bytes to the staging logits
   * buffer and run the same shader as the zero-copy path.
   */
  async run(logits, params, predOut, scoresOut) {
    const dev = this.device;
    const { C, maxLen, V, numTargetTokens } = params;
    const nPos = C * numTargetTokens;
    const fp16Bytes = 2 * C * maxLen * V * 2;

    const bytesView = new Uint8Array(logits.buffer, logits.byteOffset, logits.byteLength);
    if (bytesView.byteLength < fp16Bytes) {
      throw new Error(`run: logits byteLength ${bytesView.byteLength} < expected ${fp16Bytes} (fp16)`);
    }

    this._ensureLogitsBuf(fp16Bytes);

    this._writeParams(params, {});  // B=2 defaults

    dev.queue.writeBuffer(this.logitsBuf, 0, bytesView, 0, fp16Bytes);

    const bindGroup = dev.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.logitsBuf, size: fp16Bytes } },
        { binding: 2, resource: { buffer: this.logitsBuf, size: fp16Bytes } },
        { binding: 3, resource: { buffer: this.predBuf, size: nPos * 4 } },
        { binding: 4, resource: { buffer: this.scoresBuf, size: nPos * 4 } },
      ],
    });

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(nPos / 64));
    pass.end();

    encoder.copyBufferToBuffer(this.predBuf, 0, this.predReadBuf, 0, nPos * 4);
    encoder.copyBufferToBuffer(this.scoresBuf, 0, this.scoresReadBuf, 0, nPos * 4);
    dev.queue.submit([encoder.finish()]);

    await Promise.all([
      this.predReadBuf.mapAsync(GPUMapMode.READ),
      this.scoresReadBuf.mapAsync(GPUMapMode.READ),
    ]);

    predOut.set(new Int32Array(this.predReadBuf.getMappedRange(), 0, nPos));
    scoresOut.set(new Float32Array(this.scoresReadBuf.getMappedRange(), 0, nPos));

    this.predReadBuf.unmap();
    this.scoresReadBuf.unmap();
  }

  destroy() {
    for (const k of ['logitsBuf', 'paramsBuf',
                     'predBuf', 'scoresBuf', 'predReadBuf', 'scoresReadBuf']) {
      if (this[k]) { this[k].destroy(); this[k] = null; }
    }
    if (this.device && this._ownsDevice) { this.device.destroy(); }
    this.device = null;
  }
}
