"""
whisper_internals.py
====================
Step-by-step "under the hood" functions for the pretrained OpenAI Whisper model.

Instead of calling model.transcribe() as a black box, every stage of the pipeline is
re-done here with Whisper's own building blocks, and every intermediate result is
returned so it can be shown, plotted and checked:

    audio file
      -> 16 kHz waveform                                  load_waveform()
      -> pad / trim to 30 s                               (in analyze())
      -> STFT -> power -> mel filterbank -> log -> norm    feature_steps()
      -> conv1 -> conv2(stride 2) -> + positional emb.
         -> N Transformer encoder blocks -> audio features encoder_walkthrough()
      -> language detection (one decoder step)            detect_language()
      -> autoregressive decoding, token by token          greedy_decode_steps()
      -> cross-attention: which audio frame each token
         "listens" to                                     cross_attention_alignment()

Every manual re-implementation is compared with the library's own result
(max_abs_diff / official_text) so we can PROVE we reproduced the model correctly.

Evaluation helpers (WER breakdown, noise robustness, model-size comparison) are at the bottom.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import whisper
import whisper.audio as wa
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import LANGUAGES, get_tokenizer

SAMPLE_RATE = wa.SAMPLE_RATE      # 16000 samples / second
N_FFT = wa.N_FFT                  # 400 samples  = 25 ms analysis window
HOP = wa.HOP_LENGTH               # 160 samples  = 10 ms hop
N_SAMPLES = wa.N_SAMPLES          # 480000       = 30 s
N_FRAMES = wa.N_FRAMES            # 3000 mel frames for 30 s
ENC_STEP_S = 2 * HOP / SAMPLE_RATE  # 0.02 s: one encoder position (conv2 has stride 2)


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------
def _device(model):
    return next(model.parameters()).device


@contextmanager
def sdpa_disabled():
    """Newer PyTorch uses a fused attention kernel that never exposes the attention
    scores. Switch it off so the (query x key) scores can be read out with hooks."""
    cls = whisper.model.MultiHeadAttention
    had = "use_sdpa" in cls.__dict__
    old = cls.use_sdpa if had else None
    cls.use_sdpa = False
    try:
        yield
    finally:
        if had:
            cls.use_sdpa = old
        else:
            del cls.use_sdpa


@contextmanager
def capture_qk(modules):
    """Forward hooks that store the raw attention scores (q.k / sqrt(d)) of each module."""
    store = {}
    handles = [
        m.register_forward_hook(lambda _m, _i, out, k=k: store.__setitem__(k, out[-1]))
        for k, m in enumerate(modules)
    ]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
        for k, v in list(store.items()):
            if v is None:
                raise RuntimeError("Attention scores unavailable (fused attention kernel is on).")


def display_token(text: str) -> str:
    """Make whitespace visible in tables (Whisper tokens often start with a space)."""
    return text.replace(" ", "␣") if text.strip() == "" or text.startswith(" ") else text


# --------------------------------------------------------------------------------------
# 0. the model itself
# --------------------------------------------------------------------------------------
def model_summary(model) -> dict:
    d = model.dims
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    return {
        "dims": {k: int(v) for k, v in vars(d).items()},
        "encoder_params": enc,
        "decoder_params": dec,
        "total_params": enc + dec,
        "multilingual": bool(model.is_multilingual),
        "encoder_positions": int(d.n_audio_ctx),
        "decoder_max_tokens": int(d.n_text_ctx),
    }


# --------------------------------------------------------------------------------------
# 1. audio -> waveform
# --------------------------------------------------------------------------------------
def load_waveform(path: str):
    """Decode any audio file with ffmpeg, mono, resampled to 16 kHz, float32 in [-1, 1]."""
    audio = whisper.load_audio(path)
    return audio, len(audio) / SAMPLE_RATE


# --------------------------------------------------------------------------------------
# 2. waveform -> log-Mel spectrogram, rebuilt by hand
# --------------------------------------------------------------------------------------
def feature_steps(audio_30s: np.ndarray, n_mels: int = 80, device="cpu") -> dict:
    """Re-implements whisper.log_mel_spectrogram() one line at a time."""
    x = torch.from_numpy(np.ascontiguousarray(audio_30s)).to(device)
    window = torch.hann_window(N_FFT).to(device)
    stft = torch.stft(x, N_FFT, HOP, window=window, return_complex=True)   # (201, 3001)
    power = stft[..., :-1].abs() ** 2                                      # (201, 3000)
    filters = wa.mel_filters(device, n_mels)                               # (n_mels, 201)
    mel = filters @ power                                                  # (n_mels, 3000)
    log = torch.clamp(mel, min=1e-10).log10()
    log_floor = torch.maximum(log, log.max() - 8.0)                        # keep 80 dB range
    final = (log_floor + 4.0) / 4.0                                        # roughly [-1, 1.5]
    reference = whisper.log_mel_spectrogram(x, n_mels=n_mels)              # library result
    return {
        "window": window.cpu().numpy(),
        "power": power.cpu().numpy(),
        "power_db": (10 * torch.clamp(power, min=1e-10).log10()).cpu().numpy(),
        "filters": filters.cpu().numpy(),
        "mel": mel.cpu().numpy(),
        "log": log.cpu().numpy(),
        "final": final.cpu().numpy(),
        "final_tensor": final,
        "max_abs_diff_vs_library": float((final - reference).abs().max()),
        "n_fft": N_FFT,
        "hop": HOP,
        "n_mels": n_mels,
        "freq_bins": int(power.shape[0]),
        "frames": int(power.shape[1]),
    }


# --------------------------------------------------------------------------------------
# 3. encoder, stage by stage
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encoder_walkthrough(model, mel: torch.Tensor) -> dict:
    """Runs AudioEncoder.forward() by hand and records every intermediate tensor."""
    enc = model.encoder
    dev = _device(model)
    x = mel.unsqueeze(0).to(dev)  # (1, n_mels, 3000)
    stages = []

    def rec(name, t, params=0, note=""):
        stages.append({
            "stage": name,
            "shape": " x ".join(str(s) for s in t.shape),
            "mean": float(t.mean()),
            "std": float(t.std()),
            "params": int(params),
            "what it does": note,
        })

    rec("input log-Mel", x, 0, "80 mel bands x 3000 frames (10 ms each) = 30 s")
    h = F.gelu(enc.conv1(x))
    rec("conv1 + GELU (k=3, stride 1)", h, sum(p.numel() for p in enc.conv1.parameters()),
        "local pattern detector, mel bands -> d_model channels")
    h = F.gelu(enc.conv2(h))
    rec("conv2 + GELU (k=3, stride 2)", h, sum(p.numel() for p in enc.conv2.parameters()),
        "halves the time axis: 3000 frames -> 1500 positions (20 ms each)")
    h = h.permute(0, 2, 1)
    rec("transpose -> (batch, time, channels)", h, 0, "sequence of 1500 vectors")
    h = h + enc.positional_embedding
    rec("+ sinusoidal positional embedding", h, 0, "tells the model WHEN each vector occurred")

    layers = []
    for i, block in enumerate(enc.blocks):
        prev = h
        h = block(h)
        n_par = sum(p.numel() for p in block.parameters())
        rec(f"encoder block {i + 1}", h, n_par, "self-attention + MLP (residual, pre-LayerNorm)")
        layers.append({
            "block": i + 1,
            "cosine similarity to previous layer": float(F.cosine_similarity(prev, h, dim=-1).mean()),
            "relative change |out-in| / |in|": float(((h - prev).norm(dim=-1) / prev.norm(dim=-1)).mean()),
        })
    out = enc.ln_post(h)
    rec("final LayerNorm = audio features", out, sum(p.numel() for p in enc.ln_post.parameters()),
        "this tensor is what the decoder listens to")

    library = enc(x)
    return {
        "stages": stages,
        "layers": layers,
        "audio_features": out,                       # tensor (1, 1500, d_model)
        "audio_features_np": out[0].cpu().numpy(),
        "positional_embedding": enc.positional_embedding.cpu().numpy(),
        "max_abs_diff_vs_library": float((out - library).abs().max()),
    }


@torch.no_grad()
def encoder_self_attention(model, mel: torch.Tensor, layer: int, head: int, n_real_enc: int):
    """Attention pattern of one encoder head: how much each audio position looks at the others."""
    dev = _device(model)
    attn_module = model.encoder.blocks[layer].attn
    with sdpa_disabled(), capture_qk([attn_module]) as store:
        model.encoder(mel.unsqueeze(0).to(dev))
    w = torch.softmax(store[0][0, head], dim=-1)  # (1500, 1500)
    n = int(min(max(n_real_enc, 1), w.shape[0]))
    return w[:n, :n].cpu().numpy()


# --------------------------------------------------------------------------------------
# 4. language identification (one decoder step)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def detect_language(model, mel: torch.Tensor, top: int = 5):
    if not model.is_multilingual:
        return [("en", 1.0, "English")]
    tok = get_tokenizer(True, num_languages=model.num_languages)
    _, probs = model.detect_language(mel.to(_device(model)), tok)
    items = sorted(probs.items(), key=lambda kv: -kv[1])[:top]
    return [(code, float(p), LANGUAGES[code].title()) for code, p in items]


# --------------------------------------------------------------------------------------
# 5. decoder: token-by-token greedy decoding with probabilities
# --------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode_steps(model, audio_features: torch.Tensor, language: str = "en",
                        task: str = "transcribe", max_steps: int = 224, top_k: int = 5) -> dict:
    """Our own decoding loop. Each step: run the decoder on the tokens so far, softmax the
    logits of the last position, and pick the most probable token (greedy)."""
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                        language=language if model.is_multilingual else None,
                        task=task if model.is_multilingual else None)
    prompt = list(tok.sot_sequence_including_notimestamps)
    n_vocab = model.dims.n_vocab

    # same suppression rules the library applies (so results are comparable)
    suppress = set(tok.non_speech_tokens) | {tok.transcribe, tok.translate, tok.sot,
                                             tok.sot_prev, tok.sot_lm}
    if tok.no_speech is not None:
        suppress.add(tok.no_speech)
    mask = torch.zeros(n_vocab, device=audio_features.device)
    mask[[t for t in suppress if t < n_vocab]] = float("-inf")
    mask[tok.timestamp_begin:] = float("-inf")           # we decode without timestamps
    blank_ids = list(tok.encode(" ")) + [tok.eot]

    tokens = list(prompt)
    steps = []
    max_steps = max(1, min(max_steps, model.dims.n_text_ctx - len(prompt) - 1))
    # key/value cache: attention keys & values of earlier tokens are stored, so after the
    # first step only the newest token has to be pushed through the decoder.
    kv_cache, hooks = model.install_kv_cache_hooks()
    try:
        for step in range(max_steps):
            feed = tokens if step == 0 else tokens[-1:]
            t = torch.tensor([feed], device=audio_features.device)
            logits = model.decoder(t, audio_features, kv_cache=kv_cache)[0, -1] + mask
            if step == 0:
                logits[blank_ids] = float("-inf")         # never start with a blank
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, top_k)
            nxt = int(top_i[0])
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            steps.append({
                "step": step + 1,
                "chosen": display_token(tok.decode([nxt])),
                "probability": float(top_p[0]),
                "entropy (nats)": entropy,
                "runner-ups": ", ".join(
                    f"{display_token(tok.decode([int(i)]))} ({float(p) * 100:.1f}%)"
                    for i, p in zip(top_i[1:], top_p[1:])
                ),
                "token_id": nxt,
            })
            if nxt == tok.eot:
                break
            tokens.append(nxt)
    finally:
        for h in hooks:
            h.remove()

    text_tokens = [t for t in tokens[len(prompt):] if t < tok.eot]
    return {
        "prompt_tokens": prompt,
        "prompt_text": [tok.decode([t]) for t in prompt],
        "text_tokens": text_tokens,
        "text": tok.decode(text_tokens).strip(),
        "steps": steps,
        "tokenizer": tok,
        "ended_with_eot": bool(steps and steps[-1]["token_id"] == tok.eot),
    }


# --------------------------------------------------------------------------------------
# 6. cross-attention: where in the audio does each token look?
# --------------------------------------------------------------------------------------
@torch.no_grad()
def cross_attention_alignment(model, audio_features: torch.Tensor, decode: dict, n_samples: int) -> dict:
    """Teacher-forces the decoded tokens once more and reads the decoder->encoder attention
    of Whisper's 'alignment heads'. This is the same signal Whisper uses for word timestamps."""
    tok = decode["tokenizer"]
    prompt, text_tokens = decode["prompt_tokens"], decode["text_tokens"]
    seq = prompt + text_tokens + [tok.eot]
    t = torch.tensor([seq], device=audio_features.device)
    blocks = [b.cross_attn for b in model.decoder.blocks]
    with sdpa_disabled(), capture_qk(blocks) as store:
        model.decoder(t, audio_features)

    n_enc = int(min(max(np.ceil(n_samples / (2 * HOP)), 1), audio_features.shape[1]))
    heads = model.alignment_heads.indices().T.tolist()
    w = torch.stack([store[l][0, h][:, :n_enc] for l, h in heads])          # (heads, tokens, n_enc)
    w = torch.softmax(w, dim=-1)                                            # re-normalise on real audio only
    avg = w.mean(0)
    # row p of the matrix is the attention used to PREDICT token p+1
    first = len(prompt) - 1
    matrix = avg[first: len(seq) - 1].cpu().numpy()                         # (len(text)+1, n_enc)
    labels = [display_token(tok.decode([x])) for x in text_tokens] + ["<end>"]
    peak_s = matrix.argmax(axis=1) * ENC_STEP_S
    return {
        "matrix": matrix,
        "labels": labels,
        "peak_seconds": peak_s,
        "n_heads_used": len(heads),
        "n_enc": n_enc,
        "seconds": n_enc * ENC_STEP_S,
    }


# --------------------------------------------------------------------------------------
# 7. everything in one call (what the app runs when you press "Analyse")
# --------------------------------------------------------------------------------------
def analyze(model, audio_path: str, language: str | None = None, top_k: int = 5) -> dict:
    dev = _device(model)
    timing = {}

    t0 = time.perf_counter()
    audio, duration = load_waveform(audio_path)
    audio_30 = whisper.pad_or_trim(audio)
    timing["1. load + resample to 16 kHz"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    feats = feature_steps(audio_30, model.dims.n_mels, dev)
    mel = feats["final_tensor"]
    timing["2. log-Mel features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc = encoder_walkthrough(model, mel)
    timing["3. encoder (Transformer)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    langs = detect_language(model, mel)
    lang_code = language or langs[0][0]
    timing["4. language detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = greedy_decode_steps(model, enc["audio_features"], lang_code, top_k=top_k)
    timing["5. decoder (token by token)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    attn = cross_attention_alignment(model, enc["audio_features"], dec, len(audio))
    timing["6. cross-attention read-out"] = time.perf_counter() - t0

    # ground truth check: the library's own single-window decode with the same settings
    opts = whisper.DecodingOptions(language=lang_code if model.is_multilingual else None,
                                   without_timestamps=True, fp16=False)
    official = whisper.decode(model, mel.to(dev), opts).text.strip()

    return {
        "audio": audio, "audio_30": audio_30, "duration": duration, "n_samples": len(audio),
        "features": feats, "mel": mel, "encoder": enc, "languages": langs,
        "language_used": lang_code, "decode": dec, "attention": attn,
        "official_text": official,
        "matches_official": _same(dec["text"], official),
        "timing": timing, "n_real_enc": attn["n_enc"],
    }


def _same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


# --------------------------------------------------------------------------------------
# 8. evaluation helpers
# --------------------------------------------------------------------------------------
_NORM = None


def normalise(text: str) -> str:
    global _NORM
    if _NORM is None:
        _NORM = EnglishTextNormalizer()
    return _NORM(text)


def wer_breakdown(reference: str, hypothesis: str, normalize: bool = True) -> dict:
    """WER = (S + D + I) / N, plus the exact word-level alignment."""
    r = normalise(reference) if normalize else reference.lower().strip()
    h = normalise(hypothesis) if normalize else hypothesis.lower().strip()
    if not r.split():
        raise ValueError("reference is empty")
    if not h.split():
        h = "<empty>"   # jiwer cannot score an empty hypothesis; every word counts as deleted
    out = jiwer.process_words(r, h)
    ops = []
    for c in out.alignments[0]:
        rw = out.references[0][c.ref_start_idx:c.ref_end_idx]
        hw = out.hypotheses[0][c.hyp_start_idx:c.hyp_end_idx]
        ops.append({"operation": c.type, "reference": " ".join(rw), "model output": " ".join(hw)})
    return {
        "wer": float(out.wer), "substitutions": int(out.substitutions),
        "deletions": int(out.deletions), "insertions": int(out.insertions),
        "hits": int(out.hits), "n_ref_words": len(out.references[0]),
        "ref_normalised": r, "hyp_normalised": h, "ops": ops,
    }


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add white Gaussian noise so that signal power / noise power = snr_db."""
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), audio.shape)
    return (audio + noise).astype(np.float32)


def noise_experiment(model, audio: np.ndarray, reference: str, language: str | None,
                     snrs=(None, 20, 10, 5, 0), progress=None) -> list[dict]:
    rows = []
    for i, snr in enumerate(snrs):
        a = audio if snr is None else add_noise(audio, snr)
        t0 = time.perf_counter()
        res = model.transcribe(a, fp16=False, language=language, temperature=0.0, verbose=None)
        dt = time.perf_counter() - t0
        text = res["text"].strip()
        w = wer_breakdown(reference, text)
        rows.append({"SNR (dB)": "clean" if snr is None else str(snr), "transcript": text,
                     "WER %": round(w["wer"] * 100, 1), "S": w["substitutions"],
                     "D": w["deletions"], "I": w["insertions"], "seconds": round(dt, 2)})
        if progress:
            progress((i + 1) / len(snrs))
    return rows


def compare_models(names, loader, audio: np.ndarray, reference: str, language: str | None,
                   progress=None) -> list[dict]:
    rows = []
    for i, name in enumerate(names):
        m = loader(name)
        t0 = time.perf_counter()
        res = m.transcribe(audio, fp16=False, language=language, temperature=0.0, verbose=None)
        dt = time.perf_counter() - t0
        text = res["text"].strip()
        w = wer_breakdown(reference, text)
        rows.append({"model": name, "parameters (M)": round(model_summary(m)["total_params"] / 1e6, 1),
                     "transcript": text, "WER %": round(w["wer"] * 100, 1),
                     "S": w["substitutions"], "D": w["deletions"], "I": w["insertions"],
                     "seconds": round(dt, 2)})
        if progress:
            progress((i + 1) / len(names))
    return rows


def compare_decoding(model, audio: np.ndarray, reference: str, language: str | None,
                     beam_size: int = 5) -> list[dict]:
    rows = []
    for name, kw in [("greedy (always take the best token)", {}),
                     (f"beam search (keep {beam_size} candidate sentences)", {"beam_size": beam_size})]:
        t0 = time.perf_counter()
        res = model.transcribe(audio, fp16=False, language=language, temperature=0.0,
                               verbose=None, **kw)
        dt = time.perf_counter() - t0
        text = res["text"].strip()
        w = wer_breakdown(reference, text)
        rows.append({"strategy": name, "transcript": text, "WER %": round(w["wer"] * 100, 1),
                     "seconds": round(dt, 2)})
    return rows


def word_timestamps(model, audio: np.ndarray, language: str | None) -> list[dict]:
    """Whisper's own word timings (dynamic time warping over the same cross-attention heads)."""
    res = model.transcribe(audio, fp16=False, language=language, temperature=0.0,
                           word_timestamps=True, verbose=None)
    out = []
    for seg in res["segments"]:
        for w in seg.get("words", []):
            out.append({"word": w["word"].strip(), "start (s)": round(float(w["start"]), 2),
                        "end (s)": round(float(w["end"]), 2),
                        "confidence": round(float(w["probability"]), 3)})
    return out
