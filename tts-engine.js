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

// ─── B=1 KV main model: architecture constants ─────────────────────────────
// Coupled to the Qwen3 LLM used by OmniVoice (see omnivoice-kv-export/). The
// ONNX graph has 28 past_key_i/past_value_i inputs and the matching present_*_i
// outputs. The production export is the fp16 B=1 KV graph.
const KV_NUM_LAYERS = 28;
const KV_HEADS = 8;
const KV_HEAD_DIM = 128;
// Production main model. The bridge is intentionally standardized on this one
// fp16 B=1 KV split artifact.
const MAIN_MODEL_FILE = 'omnivoice-main-kv-fp16-b1.onnx';
const MAIN_MODEL_MANIFEST = 'omnivoice-main-kv-fp16-b1-manifest.json';
const MAIN_MODEL_VARIANT = 'kv-fp16-b1';
const DEFAULT_MODEL_REPO_ID = 'MarkShark2/omnivoice-onnx-kv-b1-fp16';
const DEFAULT_MODEL_BASE_URL = 'https://huggingface.co/MarkShark2/omnivoice-onnx-kv-b1-fp16/resolve/main';

// ─── ORT-WebGPU correctness diagnostics ────────────────────────────────────
// Keep all false for the normal fast path. These toggles isolate correctness
// regressions seen on newer ORT-WebGPU builds without changing model export.
const DIAG_FORCE_CPU_STAGED_POSTPROC = false;
const DIAG_FORCE_JS_POSTPROC = false;
const DIAG_FORCE_WASM_DECODER = false;
const DIAG_FORCE_CPU_KV_CACHE = false;

let activeRuntime = {
  mainEp: 'webgpu',
  decoderEp: 'webgpu',
  encoderEp: 'webgpu',
};
let mainSessionVariant = MAIN_MODEL_VARIANT;
let decoderSessionInfo = null;

// ─── Logging ────────────────────────────────────────────────────────────────

function log(stage, detail) {
  const msg = `[${stage}] ${detail}`;
  console.log(msg);
}

function errorDetail(err) {
  if (!err) return String(err);
  const parts = [];
  if (err.name) parts.push(err.name);
  if (err.message) parts.push(err.message);
  if (!parts.length) parts.push(String(err));
  if (err.stack) parts.push(err.stack);
  return parts.join(': ');
}

async function fetchJsonChecked(url, label) {
  const response = await fetch(url);
  const body = await response.text();
  if (!response.ok) {
    const preview = body ? ` body=${body.slice(0, 200)}` : '';
    throw new Error(`Failed to fetch ${label}: HTTP ${response.status} ${response.statusText} from ${url}.${preview}`);
  }
  try {
    return JSON.parse(body);
  } catch (err) {
    throw new Error(`Invalid JSON for ${label} from ${url}: ${err.message}; body=${body.slice(0, 200)}`);
  }
}

// ─── Tensor helper ──────────────────────────────────────────────────────────

function T(type, data, dims) { return new ort.Tensor(type, data, dims); }

async function tensorData(tensor) {
  if (typeof tensor?.getData === 'function') return await tensor.getData();
  if (tensor?.data && typeof tensor.data.slice === 'function') return tensor.data.slice();
  if (tensor?.data) return tensor.data;
  throw new Error('tensor has no getData() or data field');
}

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

const _g = new Float32Array(1025);

function normalizeFp16LogitsToF32(logits) {
  if (typeof Float16Array !== 'undefined' && logits instanceof Float16Array) {
    return new Float32Array(logits);
  }
  if (logits instanceof Uint16Array) {
    return new Float32Array(new Float16Array(logits.buffer, logits.byteOffset, logits.length));
  }
  return logits;
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
      let mx = -Infinity;
      for (let v = 0; v < V; v++) {
        const gv = gScale1 * condLogits[cOff + v] - guidanceScale * uncondLogits[uOff + v];
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

  // ── B=1 KV split state ──────────────────────────────────────────────────
  // Cond sees the full prefix. Uncond only sees the target window.
  const b1CondIds = new BigInt64Array(C * maxLen);
  for (let c = 0; c < C; c++)
    for (let s = 0; s < maxLen; s++)
      b1CondIds[c * maxLen + s] = inputIds[c * totalLen + s];
  const b1CondMask = new Uint8Array(maxLen);
  for (let s = 0; s < maxLen; s++) b1CondMask[s] = audioMask[s];
  const b1CondAttn = new Uint8Array(maxLen * maxLen);
  for (let q = 0; q < maxLen; q++)
    for (let k = 0; k < maxLen; k++)
      b1CondAttn[q * maxLen + k] = 1;
  const b1CondPos = new BigInt64Array(maxLen);
  for (let s = 0; s < maxLen; s++) b1CondPos[s] = BigInt(s);

  const b1UncondIds = new BigInt64Array(C * uncondLen).fill(BigInt(maskId));
  const b1UncondMask = new Uint8Array(uncondLen).fill(1);
  const b1UncondAttn = new Uint8Array(uncondLen * uncondLen);
  for (let q = 0; q < uncondLen; q++)
    for (let k = 0; k < uncondLen; k++)
      b1UncondAttn[q * uncondLen + k] = 1;
  const b1UncondPos = new BigInt64Array(uncondLen);
  for (let s = 0; s < uncondLen; s++) b1UncondPos[s] = BigInt(s);

  const b1StepCondIds = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
  const b1StepCondMask = new Uint8Array(numTargetTokens).fill(1);
  const b1StepCondAttn = new Uint8Array(numTargetTokens * maxLen);
  for (let q = 0; q < numTargetTokens; q++)
    for (let k = 0; k < maxLen; k++)
      b1StepCondAttn[q * maxLen + k] = 1;
  const b1StepCondPos = new BigInt64Array(numTargetTokens);
  for (let t = 0; t < numTargetTokens; t++) b1StepCondPos[t] = BigInt(targetOff + t);

  const b1StepUncondIds = new BigInt64Array(C * numTargetTokens).fill(BigInt(maskId));
  const b1StepUncondMask = new Uint8Array(numTargetTokens).fill(1);
  const b1StepUncondAttn = new Uint8Array(numTargetTokens * uncondLen);
  for (let q = 0; q < numTargetTokens; q++)
    for (let k = 0; k < uncondLen; k++)
      b1StepUncondAttn[q * uncondLen + k] = 1;
  const b1StepUncondPos = new BigInt64Array(numTargetTokens);
  for (let t = 0; t < numTargetTokens; t++) b1StepUncondPos[t] = BigInt(t);

  const b1PrefixCondZeros = {
    zeros: new Float16Array(KV_HEADS * maxLen * KV_HEAD_DIM),
    shape: [1, KV_HEADS, maxLen, KV_HEAD_DIM],
  };
  const b1PrefixUncondZeros = {
    zeros: new Float16Array(KV_HEADS * uncondLen * KV_HEAD_DIM),
    shape: [1, KV_HEADS, uncondLen, KV_HEAD_DIM],
  };

  let frozenPresentKvCond = null, frozenPresentKvUncond = null;
  let uncondRunsExecuted = 0;

  let totalInferenceMs = 0, totalModelMs = 0, totalPostProcMs = 0, totalSampleMs = 0;
  let generationMainSource = activeRuntime.mainEp;
  window._abortRequested = false;
  for (let step = 0; step < numStep; step++) {
    if (window._abortRequested) {
      log('generating', 'Generation aborted by client signal.');
      throw new Error('ABORTED_BY_CLIENT');
    }
    const k = sched[step];
    if (k <= 0) continue;
    log('generating', `Running step ${step + 1}/${numStep}...`);

    const stepT0 = performance.now();

    const modelT0 = performance.now();

    // ── Build per-step B=1 split input tensors ───────────────────────────

    const nPos = C * numTargetTokens;
    const pred = step === 0 ? new Int32Array(nPos) : pred_buf;
    const scores = step === 0 ? new Float32Array(nPos) : scores_buf;
    if (step === 0) { pred_buf = pred; scores_buf = scores; }

    {
      let condTensors;
      let uncondTensors;
      let condLogitsSeqLen, condLogitsTargetOff, uncondLogitsSeqLen;

      if (step === 0) {
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
        condLogitsSeqLen = maxLen;
        condLogitsTargetOff = targetOff;
        uncondLogitsSeqLen = uncondLen;
      } else {
        for (let c = 0; c < C; c++) {
          for (let t = 0; t < numTargetTokens; t++) {
            const v = tokens[c * numTargetTokens + t];
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
        condLogitsSeqLen = numTargetTokens;
        condLogitsTargetOff = 0;
        uncondLogitsSeqLen = numTargetTokens;
      }

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

      const getTensorCpu = async (tensor) => tensorData(tensor);

      for (const name of Object.keys(condTensors)) {
        if (step > 0 && (name.startsWith('past_key_') || name.startsWith('past_value_'))) continue;
        condTensors[name].dispose();
      }
      for (const name of Object.keys(uncondTensors)) {
        if (step > 0 && (name.startsWith('past_key_') || name.startsWith('past_value_'))) continue;
        uncondTensors[name].dispose();
      }

      if (step === 0) {
        frozenPresentKvCond = new Array(2 * KV_NUM_LAYERS);
        frozenPresentKvUncond = new Array(2 * KV_NUM_LAYERS);
        for (let i = 0; i < KV_NUM_LAYERS; i++) {
          frozenPresentKvCond[i] = condResults[`present_key_${i}`];
          frozenPresentKvCond[i + KV_NUM_LAYERS] = condResults[`present_value_${i}`];
          frozenPresentKvUncond[i] = uncondResults[`present_key_${i}`];
          frozenPresentKvUncond[i + KV_NUM_LAYERS] = uncondResults[`present_value_${i}`];
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

      const ppParams = {
        C, V, numTargetTokens, maskId, guidanceScale, layerPenalty,
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
          await gpuPostProc.runGpuSplit(
            condLogitsTensor.gpuBuffer, uncondLogitsTensor.gpuBuffer,
            ppParams, ppExt,
            pred, scores,
          );
          usedGpu = true;
        } catch (e) {
          log('warning', `B=1 split zero-copy post-proc failed (${e.message}); disabling zero-copy and falling back.`);
          gpuPostProc.sharedWithOrt = false;
        }
      }
      if (!usedGpu && (DIAG_FORCE_JS_POSTPROC || !gpuPostProc)) {
        const condData = await getTensorCpu(condLogitsTensor);
        const uncondData = await getTensorCpu(uncondLogitsTensor);
        cpuPostProcessSplit(
          condData, uncondData,
          C, condLogitsSeqLen, condLogitsTargetOff,
          uncondLogitsSeqLen, 0,
          V, numTargetTokens, maskId, guidanceScale, layerPenalty,
          pred, scores,
        );
      } else if (!usedGpu) {
        const condData = await getTensorCpu(condLogitsTensor);
        const uncondData = await getTensorCpu(uncondLogitsTensor);
        const condElems = C * condLogitsSeqLen * V;
        const combined = new Uint16Array(condElems + condElems);
        combined.set(new Uint16Array(condData.buffer, condData.byteOffset, condElems), 0);
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

    totalSampleMs += performance.now() - sampleT0;

    const stepMs = performance.now() - stepT0;
    totalInferenceMs += stepMs;
    log('generating', `Step ${step + 1}/${numStep} (${stepMs.toFixed(0)}ms)`);

    if (step % 5 === 0 && globalThis.gc) {
      globalThis.gc();
    }
  }

  // Release frozen-prefix GPU buffers now that the loop is done. Without this
  // each utterance would leak hundreds of MB of VRAM per side.
  if (frozenPresentKvCond) {
    for (const t of frozenPresentKvCond) { try { t.dispose(); } catch (_e) {} }
    frozenPresentKvCond = null;
  }
  if (frozenPresentKvUncond) {
    for (const t of frozenPresentKvUncond) { try { t.dispose(); } catch (_e) {} }
    frozenPresentKvUncond = null;
  }

  const otherMs = Math.max(0, totalInferenceMs - totalModelMs - totalPostProcMs - totalSampleMs);
  log('perf', `${numStep} steps in ${totalInferenceMs.toFixed(0)}ms total | model: ${totalModelMs.toFixed(0)}ms (${(totalModelMs / Math.max(1, numStep)).toFixed(0)}ms/step) | postproc+sync: ${totalPostProcMs.toFixed(0)}ms | sample: ${totalSampleMs.toFixed(0)}ms | other: ${otherMs.toFixed(0)}ms | mode: kv-b1-split | variant: ${mainSessionVariant} | main: ${generationMainSource} | uncond runs: ${uncondRunsExecuted}/${numStep}`);

  return tokens;
}

// ─── Decode tokens to audio ─────────────────────────────────────────────────

async function decodeTokens(tokens, C, T) {
  log('decoding', 'Converting tokens to audio...');
  const codes = new BigInt64Array(C * T);
  codes.set(tokens);
  const inTensor = new ort.Tensor('int64', codes, [1, C, T]);
  let result = null;
  try {
    result = await decoderSession.run({ audio_codes: inTensor });
    if (typeof result.audio_values.getData === 'function') {
      return await result.audio_values.getData();
    }
    return result.audio_values.data;
  } finally {
    inTensor.dispose();
    if (result?.audio_values) result.audio_values.dispose();
  }
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

async function ensureEncoderSession() {
  if (encoderSession) return encoderSession;

  log('downloading', 'Downloading encoder.onnx (654MB)...');
  try {
    const encResp = await fetch(`${window._modelBaseUrl}/omnivoice-encoder-fixed.onnx`);
    if (!encResp.ok) throw new Error(`fetch status ${encResp.status}`);
    const encBuf = await encResp.arrayBuffer();
    const encData = new Uint8Array(encBuf);

    if (activeRuntime.encoderEp === 'wasm') {
      log('loading', 'Creating encoder session (WASM)...');
      encoderSession = await ort.InferenceSession.create(encData, {
        executionProviders: ['wasm'],
        enableMemPattern: false,
        enableCpuMemArena: false,
      });
      log('init', 'Encoder loaded on WASM.');
      return encoderSession;
    }

    try {
      log('loading', 'Creating encoder session (WebGPU)...');
      encoderSession = await ort.InferenceSession.create(encData, {
        executionProviders: ['webgpu'],
        enableMemPattern: false,
        enableCpuMemArena: false,
      });
      log('init', 'Encoder loaded on WebGPU.');
    } catch (e) {
      log('warning', `Encoder WebGPU session failed (${errorDetail(e)}); falling back to WASM.`);
      encoderSession = await ort.InferenceSession.create(encData, {
        executionProviders: ['wasm'],
        enableMemPattern: false,
        enableCpuMemArena: false,
      });
      log('init', 'Encoder loaded on WASM (fallback).');
    }
    return encoderSession;
  } catch (e) {
    log('error', `Failed to load encoder: ${errorDetail(e)}`);
    throw e;
  }
}

// ─── Init — load models and tokenizer ───────────────────────────────────────

async function init(modelBaseUrl = DEFAULT_MODEL_BASE_URL) {
  try {
    activeRuntime = {
      mainEp: 'webgpu',
      decoderEp: 'webgpu',
      encoderEp: 'webgpu',
    };

    let hasWorkingGPU = false;
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        hasWorkingGPU = !!adapter;
        if (adapter) {
          try {
            const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
            log('init', `WebGPU adapter: ${info.vendor || '?'} ${info.device || '?'} (${info.description || '?'})`);
          } catch (_e) {
            log('init', 'WebGPU adapter found (could not read info)');
          }
        }
      } catch (e) {
        log('init', `WebGPU probe failed: ${errorDetail(e)}`);
      }
    }
    if (!hasWorkingGPU) {
      throw new Error('WebGPU is not available. This bridge now targets the AMD BC-250 WebGPU/Vulkan runtime only.');
    }

    // Provide modelBaseUrl for lazy loading
    window._modelBaseUrl = modelBaseUrl;

    // Load config
    log('loading', 'Loading config...');
    config = await fetchJsonChecked(`${modelBaseUrl}/omnivoice-config.json`, 'omnivoice-config.json');
    log('loading', `Config loaded: ${config.num_audio_codebook} codebooks, vocab ${config.audio_vocab_size}, sr ${config.sampling_rate}`);

    // Load tokenizer
    log('loading', `Loading tokenizer (Qwen2 BPE) from ${DEFAULT_MODEL_REPO_ID}...`);
    tokenizer = await AutoTokenizer.from_pretrained(DEFAULT_MODEL_REPO_ID);
    log('loading', 'Tokenizer loaded.');

    // Whisper ASR is now lazy-loaded natively when new voices require pre-computation.

    const mainModelFile = MAIN_MODEL_FILE;
    const manifestFile = MAIN_MODEL_MANIFEST;
    mainSessionVariant = MAIN_MODEL_VARIANT;
    log('loading', `Using production main-model variant: ${mainSessionVariant}.`);

    // Load model data shards
    log('downloading', `Loading main model shards into memory (${manifestFile})...`);
    const dataFiles = await fetchJsonChecked(`${modelBaseUrl}/${manifestFile}`, manifestFile);
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
    let actualBackend = 'webgpu';
    log('loading', 'Creating main model session (WebGPU)...');

    const opt = {
      externalData: externalData,
      graphOptimizationLevel: 'all',
      enableMemPattern: false,
      enableCpuMemArena: false,
      executionProviders: ['webgpu'],
      preferredOutputLocation: {},
    };
    if (!DIAG_FORCE_CPU_STAGED_POSTPROC && !DIAG_FORCE_JS_POSTPROC) {
      opt.preferredOutputLocation = { audio_logits: 'gpu-buffer' };
    }
    if (!DIAG_FORCE_CPU_KV_CACHE) {
      for (let i = 0; i < KV_NUM_LAYERS; i++) {
        opt.preferredOutputLocation[`present_key_${i}`] = 'gpu-buffer';
        opt.preferredOutputLocation[`present_value_${i}`] = 'gpu-buffer';
      }
    }

    try {
      mainSession = await ort.InferenceSession.create(`${modelBaseUrl}/${mainModelFile}`, opt);
    } catch (e) {
      log('error', `Main model WebGPU failed: ${errorDetail(e)}`);
      throw new Error(`WebGPU InferenceSession creation failed: ${e.message || String(e)}`);
    }

    // CRITICAL: Free the JavaScript ArrayBuffers so the browser doesn't swap + lock up!
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
      log('init', `Initialized GpuPostProcessor (zero-copy logits: ${ortDevice ? 'on' : 'off - ORT device not accessible'})`);
    }

    const loadDecoderWasm = async () => {
      log('loading', 'Downloading decoder for WASM...');
      let decName = 'omnivoice-decoder-webgpu.onnx';
      let decResp = await fetch(`${modelBaseUrl}/${decName}`);
      if (!decResp.ok) {
        decName = 'omnivoice-decoder.onnx';
        decResp = await fetch(`${modelBaseUrl}/${decName}`);
      }
      if (!decResp.ok) throw new Error(`fetch ${decName} failed: ${decResp.status}`);
      const decBuf = await decResp.arrayBuffer();
      decoderSession = await ort.InferenceSession.create(new Uint8Array(decBuf), {
        executionProviders: ['wasm'],
        enableMemPattern: false,
        enableCpuMemArena: false,
      });
      decoderSessionInfo = { ep: 'wasm', model: decName, bytes: decBuf.byteLength };
    };

    if (activeRuntime.decoderEp === 'wasm') {
      await loadDecoderWasm();
      log('loading', `Decoder session created on WASM (${decoderSessionInfo.model}).`);
    } else {
      try {
        if (DIAG_FORCE_WASM_DECODER) throw new Error('DIAG_FORCE_WASM_DECODER=true');
        log('loading', 'Trying WebGPU-patched decoder (omnivoice-decoder-webgpu.onnx, ~111MB)...');
        const decResp = await fetch(`${modelBaseUrl}/omnivoice-decoder-webgpu.onnx`);
        if (!decResp.ok) throw new Error(`fetch status ${decResp.status}`);
        const decBuf = await decResp.arrayBuffer();
        decoderSession = await ort.InferenceSession.create(new Uint8Array(decBuf), {
          executionProviders: ['webgpu'],
          enableMemPattern: false,
          enableCpuMemArena: false,
        });
        decoderSessionInfo = { ep: 'webgpu', model: 'omnivoice-decoder-webgpu.onnx', bytes: decBuf.byteLength };
        log('loading', 'Decoder session created on WebGPU.');
      } catch (e) {
        log('loading', `WebGPU decoder unavailable (${errorDetail(e)}); falling back to WASM.`);
        await loadDecoderWasm();
      }
    }
    try {
      await ensureEncoderSession();
    } catch (_e) {
      log('warning', 'Encoder pre-load failed; voice cloning will retry when reference audio is used.');
    }

    log('ready', `Engine ready. Backend: ${actualBackend}`);
    return { backend: actualBackend, config, model: { repoId: DEFAULT_MODEL_REPO_ID, mainVariant: MAIN_MODEL_VARIANT } };
  } catch (err) {
    log('error', `Init failed: ${errorDetail(err)}`);
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

      await ensureEncoderSession();
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
    log('error', `Synthesis failed: ${errorDetail(err)}`);
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
    await ensureEncoderSession();
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
// The shader takes two logits storage bindings. The zero-copy path binds the
// independent cond/uncond ORT GPU buffers; the CPU-staged fallback packs those
// streams into a single staging buffer and binds it twice with an offset.
const WGSL = /* wgsl */ `

// CFG + argmax post-processing shader.
// Input:  cond/uncond logits buffers (fp16, packed as u32 pairs).
// Output: pred[nPos]   — argmax vocab id per position
//         scoresBuf[nPos] — log-prob of that argmax minus codebook penalty

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
                         //   to every uncond index when cond/uncond share a
                         //   CPU-staged buffer. Zero-copy split uses 0.
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

  // The old shader did log_softmax(cond) and log_softmax(uncond) separately:
  //
  //   gv = (1+s) * (c - c_lse) - s * (u - u_lse)
  //
  // The c_lse/u_lse part is constant over vocab and cancels when the final
  // softmax normalizes gv. So the exact same pred and scores come from the
  // fused raw logits r = (1+s)*c - s*u. This removes four full exp/log passes
  // per position, which matters on the BC-250 WebGPU/Vulkan path.
  var gMax : f32 = NEG_INF;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    let gv = gScale1 * load_fp16_cond(cBase + v) - p.guidanceScale * load_fp16_uncond(uBase + v);
    gMax = max(gMax, gv);
  }
  var gSum : f32 = 0.0;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    let gv = gScale1 * load_fp16_cond(cBase + v) - p.guidanceScale * load_fp16_uncond(uBase + v);
    gSum = gSum + exp(gv - gMax);
  }
  let gLse = gMax + log(gSum);

  var bestV : u32 = 0u;
  var bestS : f32 = NEG_INF;
  for (var v : u32 = 0u; v < V; v = v + 1u) {
    if (v == p.maskId) { continue; }
    let gv = gScale1 * load_fp16_cond(cBase + v) - p.guidanceScale * load_fp16_uncond(uBase + v);
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
    this.logitsBuf = null;        // only allocated in the CPU-staged fallback
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

  // Serialize the 12-entry Params struct into the paramsBuf.
  _writeParams(params, ext) {
    const { C, V, numTargetTokens, maskId, guidanceScale, layerPenalty } = params;
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
    *  staging buffer is only needed on the CPU-staged fallback and is allocated
    *  lazily in run().
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
   * Zero-copy path, B=1 split mode: cond and uncond logits come from two
   * independent mainSession.run() calls. Bind each buffer to its own slot and
   * set uncondBatchOff=0 so both use pure row-major indexing.
   */
  async runGpuSplit(logitsCondBuf, logitsUncondBuf, params, ext, predOut, scoresOut) {
    const dev = this.device;
    const nPos = params.C * params.numTargetTokens;

    this._writeParams(params, { ...ext, uncondBatchOff: 0 });

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
  * CPU-staged fallback: ORT-web handed us fp16 cond/uncond logits on CPU
  * (Float16Array on modern ort-web, Uint16Array on older -- both store fp16
  * bit patterns at 2 B/elem). The caller packs them as cond rows followed by
  * uncond rows, then this uploads the raw bytes and runs the same shader.
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

    this._writeParams(params, {});

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
