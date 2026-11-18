"""
Whisper Under the Hood
======================
A Streamlit app that applies the PRETRAINED OpenAI Whisper model and, instead of treating it as a
black box, shows every stage of the pipeline: waveform -> log-Mel features -> Transformer encoder
-> language detection -> token-by-token decoding -> cross-attention -> evaluation (WER, noise, model size).

Run:  streamlit run app.py
"""
import html
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import whisper

import whisper_internals as wi

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "tests" / "jfk.flac"
SAMPLE_REF = ("And so my fellow Americans, ask not what your country can do for you, "
              "ask what you can do for your country.")
LANGS = {"Auto-detect": None, "English (en)": "en", "Hindi (hi)": "hi", "Gujarati (gu)": "gu",
         "Tamil (ta)": "ta", "Telugu (te)": "te", "Spanish (es)": "es", "French (fr)": "fr",
         "German (de)": "de", "Japanese (ja)": "ja", "Chinese (zh)": "zh"}

st.set_page_config(page_title="Inside the model", page_icon="🎙️", layout="wide")


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading pretrained Whisper weights...")
def load_model(name):
    return whisper.load_model(name, device="cpu")


def save_upload(uploaded):
    suffix = os.path.splitext(getattr(uploaded, "name", "audio.wav"))[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.getvalue())
    return f.name


def safe_label(text, i):
    """Matplotlib's default font lacks Indic/CJK glyphs; fall back to the token number
    (the readable text is always in the tables)."""
    return text if all(ord(c) < 0x250 or c == "\u2423" for c in text) else f"#{i + 1}"


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ plotting helpers
def plot_waveform(audio, duration):
    t = np.arange(len(audio)) / wi.SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(t, audio, lw=0.5)
    if duration < 30:
        ax.axvspan(duration, 30, color="grey", alpha=0.25, label="zero padding added by Whisper")
        ax.legend(loc="upper right")
    ax.set_xlim(0, max(30, duration) if duration < 30 else duration)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title("Waveform: 16 000 numbers per second")
    fig.tight_layout()
    return fig


def plot_window_and_filters(feats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    axes[0].plot(feats["window"])
    axes[0].set_title(f"Hann window ({feats['n_fft']} samples = 25 ms)")
    axes[0].set_xlabel("sample")
    freqs = np.linspace(0, wi.SAMPLE_RATE / 2, feats["freq_bins"])
    for i in range(0, feats["n_mels"], 4):
        axes[1].plot(freqs, feats["filters"][i], lw=0.8)
    axes[1].set_title("Mel filterbank (every 4th of %d filters)" % feats["n_mels"])
    axes[1].set_xlabel("frequency (Hz): filters get wider at high frequency")
    fig.tight_layout()
    return fig


def plot_feature_chain(feats, duration):
    n = max(1, min(feats["frames"], int(duration * 100)))
    dur = n / 100
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    im = axes[0].imshow(feats["power_db"][:, :n], origin="lower", aspect="auto",
                        extent=[0, dur, 0, wi.SAMPLE_RATE / 2], vmin=-100)
    axes[0].set_title("(a) STFT power, dB  (201 freq bins)")
    axes[0].set_ylabel("Hz")
    fig.colorbar(im, ax=axes[0])
    mel_db = 10 * np.log10(np.maximum(feats["mel"][:, :n], 1e-10))
    im = axes[1].imshow(mel_db, origin="lower", aspect="auto", extent=[0, dur, 0, feats["n_mels"]])
    axes[1].set_title("(b) after mel filterbank, dB  (80 bands)")
    axes[1].set_ylabel("mel band")
    fig.colorbar(im, ax=axes[1])
    im = axes[2].imshow(feats["final"][:, :n], origin="lower", aspect="auto",
                        extent=[0, dur, 0, feats["n_mels"]])
    axes[2].set_title("(c) log10, floor at max-8, (x+4)/4  = model input")
    fig.colorbar(im, ax=axes[2])
    for a in axes:
        a.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def plot_features_heat(mat, seconds, title, cmap="viridis", ylabel="channel"):
    n = max(1, min(mat.shape[0], int(round(seconds / wi.ENC_STEP_S))))
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(mat[:n].T, origin="lower", aspect="auto", cmap=cmap,
                   extent=[0, n * wi.ENC_STEP_S, 0, mat.shape[1]])
    ax.set_xlabel("time (s)  (one position = 20 ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_confidence(steps, limit=60):
    s = [x for x in steps if x["chosen"] != "<|endoftext|>"][:limit]
    if not s:
        return None
    p = np.array([x["probability"] for x in s])
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(s)), 3))
    colors = np.where(p < 0.6, "#d9534f", np.where(p < 0.9, "#f0ad4e", "#5cb85c"))
    ax.bar(range(len(s)), p, color=colors)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([safe_label(x["chosen"], i) for i, x in enumerate(s)],
                       rotation=75, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(chosen token)")
    ax.set_title("Model confidence for every token it wrote (red = unsure)")
    fig.tight_layout()
    return fig


def plot_attention(att, limit=60):
    m = att["matrix"][:limit]
    labels = att["labels"][:limit]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.28 * len(labels))))
    im = ax.imshow(m, aspect="auto", extent=[0, att["seconds"], len(labels), 0], cmap="magma")
    ax.plot(att["peak_seconds"][:limit], np.arange(len(labels)) + 0.5, ".", color="cyan", ms=4)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels([safe_label(t, i) for i, t in enumerate(labels)], fontsize=8)
    ax.set_xlabel("position in the audio (s)")
    ax.set_title("Cross-attention: where each output token 'listens' (cyan = strongest)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_self_attention(w, layer, head):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(w, cmap="magma", aspect="auto", extent=[0, w.shape[1] * wi.ENC_STEP_S,
                                                           w.shape[0] * wi.ENC_STEP_S, 0])
    ax.set_xlabel("attended-to position (s)")
    ax.set_ylabel("query position (s)")
    ax.set_title(f"Encoder self-attention, block {layer + 1}, head {head + 1}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def confidence_html(steps):
    spans = []
    for s in steps:
        if s["chosen"] == "<|endoftext|>":
            continue
        p = s["probability"]
        alpha = max(0.0, min(0.9, (1 - p) * 1.3))
        txt = html.escape(s["chosen"].replace("␣", " "))
        spans.append(f"<span title='P = {p * 100:.1f}%' style='background:rgba(255,80,80,{alpha:.2f});"
                     f"padding:1px 0;white-space:pre-wrap'>{txt}</span>")
    return "<div style='font-size:1.15rem;line-height:2'>" + "".join(spans) + "</div>"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Pretrained model", ["tiny", "base", "small"], index=1,
                                  help="Same architecture, different size. Weights are downloaded once.")
lang_label = st.sidebar.selectbox("Language", list(LANGS), index=0)
top_k = st.sidebar.slider("Alternatives shown per token", 3, 8, 5)
st.sidebar.caption("Runs on CPU. Nothing is trained here: we apply OpenAI's pretrained weights "
                   "and inspect how they work.")

st.title("🔬 Inside the model: how Whisper works")
st.write("Applying a **pretrained** speech-recognition model, and opening it up to see every step "
         "from raw audio to text.")

# ------------------------------------------------------------------ input
choices = ["Upload a file", "Record with microphone"]
if SAMPLE.exists():
    choices.append("Sample clip from the repo (JFK, 11 s)")
source = st.radio("Audio source", choices, horizontal=True)
audio_in, audio_path = None, None
if source == "Upload a file":
    audio_in = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a", "flac", "ogg"])
elif source == "Record with microphone":
    if hasattr(st, "audio_input"):
        audio_in = st.audio_input("Record")
    else:
        st.warning("Recording needs a newer Streamlit: pip install -U streamlit")
else:
    audio_path = str(SAMPLE)
    st.audio(audio_path)
if audio_in is not None:
    st.audio(audio_in)

ready = audio_in is not None or audio_path is not None
if st.button("Analyse", type="primary", disabled=not ready):
    path = audio_path or save_upload(audio_in)
    model = load_model(model_name)
    with st.spinner("Running every stage of the pipeline..."):
        res = wi.analyze(model, path, language=LANGS[lang_label], top_k=top_k)
    st.session_state["analysis"] = {"res": res, "model_name": model_name,
                                    "sample": audio_path is not None}
    st.session_state.pop("noise_rows", None)
    st.session_state.pop("model_rows", None)
    st.session_state.pop("decode_rows", None)

A = st.session_state.get("analysis")
if not A:
    st.info("Pick an audio source and press **Analyse**. A good first test is the sample clip "
            "(it has a known transcript for scoring).")
    st.stop()

res, used_model = A["res"], A["model_name"]
model = load_model(used_model)
feats, enc, dec, att = res["features"], res["encoder"], res["decode"], res["attention"]
dur = res["duration"]

tabs = st.tabs(["Overview", "1 Audio", "2 Features", "3 Encoder", "4 Language", "5 Decoder",
                "6 Attention", "7 Evaluation"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.subheader("Pipeline")
    st.graphviz_chart("""
    digraph G { rankdir=LR; graph [bgcolor="white"];
      node [shape=box, style="rounded,filled", fillcolor="#e8eefc", fontname="Helvetica", fontsize=11];
      A [label="Audio file"]; B [label="16 kHz waveform\\npadded to 30 s"];
      C [label="Log-Mel spectrogram\\n80 x 3000"]; D [label="Encoder\\n2 conv + Transformer blocks\\n1500 x d_model"];
      E [label="Decoder\\ncausal self-attn + cross-attn"]; F [label="Tokens -> text"];
      A -> B -> C -> D; D -> E [label=" cross-attention "]; E -> F; F -> E [label=" next token ", style=dashed]; }
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", f"{used_model}  ({wi.model_summary(model)['total_params'] / 1e6:.0f} M parameters)")
    c2.metric("Language used", res["language_used"])
    c3.metric("Audio length", f"{dur:.1f} s")
    st.markdown("**Transcript produced by our own step-by-step decoder**")
    st.success(dec["text"] or "(empty)")
    ok = res["matches_official"]
    (st.success if ok else st.warning)(
        f"Our manual pipeline {'matches' if ok else 'differs from'} Whisper's official decode "
        f"(`whisper.decode`): “{res['official_text']}”")
    c1, c2 = st.columns(2)
    c1.metric("Manual log-Mel vs library (max abs difference)", f"{feats['max_abs_diff_vs_library']:.1e}")
    c2.metric("Manual encoder vs library (max abs difference)", f"{enc['max_abs_diff_vs_library']:.1e}")
    st.caption("A difference of ~0 proves that our re-implementation of each stage reproduces the model exactly.")
    st.subheader("Where the time goes")
    st.bar_chart(pd.Series(res["timing"], name="seconds"))

# ------------------------------------------------------------------ 1 Audio
with tabs[1]:
    st.markdown("**What happens:** the file is decoded with ffmpeg, mixed to mono and resampled to "
                "**16 000 samples per second**. Whisper always works on a **30 s window**, so shorter "
                "audio is padded with zeros (longer audio is processed in 30 s chunks).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample rate", f"{wi.SAMPLE_RATE} Hz")
    c2.metric("Samples", f"{res['n_samples']:,}")
    c3.metric("Duration", f"{dur:.2f} s")
    c4.metric("Padded length", f"{wi.N_SAMPLES:,} samples")
    show(plot_waveform(res["audio"], dur))

# ------------------------------------------------------------------ 2 Features
with tabs[2]:
    st.markdown(
        "**What happens:** a neural network cannot read raw samples well, so the waveform is turned into "
        "a picture of *which frequencies are present when*.\n"
        "1. **STFT**: slide a 25 ms Hann window (400 samples) in 10 ms steps (160 samples), Fourier-transform each "
        "slice -> 201 frequency bins per frame.\n"
        "2. **Power**: |STFT|² (energy per bin).\n"
        "3. **Mel filterbank**: 80 triangular filters spaced like human hearing (dense at low frequencies) "
        "compress 201 bins to 80 bands.\n"
        "4. **Log**: loudness is perceived logarithmically; values are floored 80 dB below the peak and "
        "rescaled to roughly [-1, 1.5].")
    show(plot_window_and_filters(feats))
    show(plot_feature_chain(feats, dur))
    st.table(pd.DataFrame({
        "quantity": ["window length", "hop", "FFT bins", "mel bands", "frames for 30 s"],
        "value": [f"{feats['n_fft']} samples (25 ms)", f"{feats['hop']} samples (10 ms)",
                  str(feats["freq_bins"]), str(feats["n_mels"]), str(feats["frames"])]}))
    st.success(f"Our hand-written version equals `whisper.log_mel_spectrogram` "
               f"(max abs difference {feats['max_abs_diff_vs_library']:.1e}).")

# ------------------------------------------------------------------ 3 Encoder
with tabs[3]:
    ms = wi.model_summary(model)
    st.markdown(
        "**What happens:** the encoder turns the 80 x 3000 log-Mel image into **1500 vectors** (one per 20 ms), "
        "each summarising what is being said at that moment *in context*. Two convolutions extract local "
        "patterns and halve the time axis; a positional embedding injects time order; then a stack of "
        "Transformer blocks lets every position look at every other position (self-attention).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Encoder parameters", f"{ms['encoder_params'] / 1e6:.1f} M")
    c2.metric("Decoder parameters", f"{ms['decoder_params'] / 1e6:.1f} M")
    c3.metric("Attention heads / layers (enc)",
              f"{ms['dims']['n_audio_head']} / {ms['dims']['n_audio_layer']}")
    st.dataframe(pd.DataFrame(enc["stages"]), hide_index=True)
    st.success(f"Our manual forward pass equals `model.encoder(...)` "
               f"(max abs difference {enc['max_abs_diff_vs_library']:.1e}).")
    st.markdown("**How much does each Transformer block change the representation?**")
    st.dataframe(pd.DataFrame(enc["layers"]), hide_index=True)
    show(plot_features_heat(enc["audio_features_np"], dur, "Encoder output (audio features), 1 row per channel"))
    show(plot_features_heat(enc["positional_embedding"], dur,
                            "Sinusoidal positional embedding added to the input (a clock signal)",
                            cmap="RdBu", ylabel="dimension"))
    with st.expander("Look inside one self-attention head of the encoder"):
        d = ms["dims"]
        c1, c2 = st.columns(2)
        layer = c1.number_input("Block", 1, d["n_audio_layer"], 1) - 1
        head = c2.number_input("Head", 1, d["n_audio_head"], 1) - 1
        if st.button("Show attention map"):
            w = wi.encoder_self_attention(model, res["mel"], int(layer), int(head), res["n_real_enc"])
            show(plot_self_attention(w, int(layer), int(head)))
            st.caption("Row = a position asking 'what should I pay attention to?'. Bright = high weight. "
                       "Early blocks tend to look at neighbours, later ones at longer-range context.")

# ------------------------------------------------------------------ 4 Language
with tabs[4]:
    st.markdown("**What happens:** the decoder is fed only `<|startoftranscript|>`. The probabilities it "
                "gives to the ~99 language tokens are the language identification. The same model does ASR "
                "and language ID because both are just 'predict the next token'.")
    ldf = pd.DataFrame(res["languages"], columns=["code", "probability", "language"])
    st.bar_chart(ldf.set_index("language")["probability"])
    st.dataframe(ldf, hide_index=True)
    st.caption(f"Language used for decoding: **{res['language_used']}**"
               + ("" if LANGS[lang_label] is None else " (set manually in the sidebar)"))

# ------------------------------------------------------------------ 5 Decoder
with tabs[5]:
    st.markdown(
        "**What happens:** the decoder is *autoregressive*. It starts from a prompt of special tokens and "
        "repeatedly (1) runs causal self-attention over the tokens so far, (2) cross-attends to the 1500 audio "
        "vectors, (3) outputs a probability for each of the ~51 865 tokens, (4) picks the best one and "
        "appends it. It stops at `<|endoftext|>`.")
    st.markdown("**Prompt tokens:** " + " ".join(f"`{t}`" for t in dec["prompt_text"])
                + "  → start, language, task, no-timestamps")
    st.markdown("**Output, shaded by the model's confidence** (hover a token for its probability):")
    st.markdown(confidence_html(dec["steps"]), unsafe_allow_html=True)
    fig = plot_confidence(dec["steps"])
    if fig is not None:
        show(fig)
    st.markdown("**Every decoding step** (low probability = the model was unsure; check the runner-ups):")
    sdf = pd.DataFrame(dec["steps"]).drop(columns=["token_id"])
    sdf["probability"] = sdf["probability"].round(4)
    sdf["entropy (nats)"] = sdf["entropy (nats)"].round(3)
    st.dataframe(sdf, hide_index=True)
    st.caption("Entropy is high when probability is spread over many tokens. This is the model's own "
               "uncertainty, which is exactly where recognition errors tend to appear.")

# ------------------------------------------------------------------ 6 Attention
with tabs[6]:
    st.markdown(
        "**What happens:** when the decoder writes a token, its *cross-attention* decides which of the 1500 "
        "audio positions to read. We read out the attention of Whisper's designated **alignment heads** "
        f"({att['n_heads_used']} heads). The bright path moving left to right shows the model really follows "
        "the audio in order, which is evidence that it is *listening* rather than guessing. "
        "Whisper uses this same signal for word-level timestamps.")
    show(plot_attention(att))
    rows = pd.DataFrame({"token": att["labels"], "peak attention at (s)": np.round(att["peak_seconds"], 2)})
    st.dataframe(rows.head(80), hide_index=True)
    if st.button("Compare with Whisper's own word timestamps (dynamic time warping)"):
        try:
            with st.spinner("Running transcription with word timestamps..."):
                st.dataframe(pd.DataFrame(wi.word_timestamps(model, res["audio"], res["language_used"])),
                             hide_index=True)
        except Exception as e:  # numba / version problems should not break the app
            st.error(f"Word timestamps unavailable: {e}")

# ------------------------------------------------------------------ 7 Evaluation
with tabs[7]:
    st.markdown("**WER = (S + D + I) / N**: substitutions + deletions + insertions, divided by the number of "
                "words in the reference. Compare the model's output with what was *actually* said.")
    default_ref = SAMPLE_REF if A["sample"] else ""
    ref = st.text_area("Reference text (what was actually said)", default_ref, key=f"ref_{A['sample']}")
    normalize = st.checkbox("Normalise text first (Whisper's EnglishTextNormalizer: numbers, spelling, punctuation)",
                            value=True)
    if ref.strip():
        w = wi.wer_breakdown(ref, dec["text"], normalize)
        c = st.columns(5)
        c[0].metric("WER", f"{w['wer'] * 100:.1f} %")
        c[1].metric("Substitutions", w["substitutions"])
        c[2].metric("Deletions", w["deletions"])
        c[3].metric("Insertions", w["insertions"])
        c[4].metric("Reference words", w["n_ref_words"])
        odf = pd.DataFrame(w["ops"])
        if not st.checkbox("Show matching words too", value=False):
            odf = odf[odf["operation"] != "equal"]
        st.dataframe(odf if len(odf) else pd.DataFrame({"result": ["No errors: perfect match"]}),
                     hide_index=True)

        st.divider()
        st.subheader("Experiment A: robustness to noise")
        st.caption("White Gaussian noise is added at a chosen signal-to-noise ratio (lower dB = noisier) "
                   "and the same audio is re-transcribed.")
        snrs = st.multiselect("SNR levels (dB)", [20, 15, 10, 5, 0, -5], default=[20, 10, 5, 0])
        if st.button("Run noise experiment") and snrs:
            bar = st.progress(0.0)
            st.session_state["noise_rows"] = wi.noise_experiment(
                model, res["audio"], ref, res["language_used"],
                snrs=[None] + sorted(snrs, reverse=True), progress=bar.progress)
        if "noise_rows" in st.session_state:
            ndf = pd.DataFrame(st.session_state["noise_rows"])
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(ndf["SNR (dB)"].astype(str), ndf["WER %"], "o-")
            ax.set_xlabel("SNR (dB), noisier ->")
            ax.set_ylabel("WER %")
            ax.set_title("WER vs noise level")
            fig.tight_layout()
            show(fig)
            st.dataframe(ndf, hide_index=True)

        st.divider()
        st.subheader("Experiment B: model size")
        st.caption("Same architecture, more layers and wider vectors. Downloads weights on first use "
                   "(tiny ≈ 75 MB, base ≈ 142 MB, small ≈ 466 MB).")
        pick = st.multiselect("Models to compare", ["tiny", "base", "small"], default=["tiny", "base"])
        if st.button("Compare models") and pick:
            bar = st.progress(0.0)
            st.session_state["model_rows"] = wi.compare_models(
                pick, load_model, res["audio"], ref, res["language_used"], progress=bar.progress)
        if "model_rows" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["model_rows"]), hide_index=True)

        st.divider()
        st.subheader("Experiment C: decoding strategy")
        st.caption("Greedy keeps only the best token at each step. Beam search keeps several candidate "
                   "sentences alive and picks the best overall.")
        if st.button("Compare greedy vs beam search"):
            st.session_state["decode_rows"] = wi.compare_decoding(model, res["audio"], ref, res["language_used"])
        if "decode_rows" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["decode_rows"]), hide_index=True)
    else:
        st.info("Type the reference text above to compute WER and unlock the experiments.")
